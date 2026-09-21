//! Checkpoint mapping and load-time layout conversion. No request computation.
use super::backend::{kernels, RuntimeBackend};
use apxinf_core::{Backend, DType, Error, Result, Tensor};
use half::bf16;
use std::{collections::HashMap, path::Path, rc::Rc};
pub(crate) struct Linear {
    pub plan: Rc<kernels::gemm::Bf16LinearPlan>,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}
pub(crate) struct Norm {
    pub weight: Tensor,
    pub bias: Tensor,
}
pub(crate) struct VisionBlock {
    pub norm1: Norm,
    pub qkv: Linear,
    pub proj: Linear,
    pub norm2: Norm,
    pub fc1: Linear,
    pub fc2: Linear,
    pub scale1: Option<Tensor>,
    pub scale2: Option<Tensor>,
}
pub(crate) struct Tower {
    pub width: usize,
    pub prefix: Option<Tensor>,
    pub position: Tensor,
    pub conv: Tensor,
    pub bias: Tensor,
    pub plan: kernels::convolution::Conv2dPlan,
    pub blocks: Vec<VisionBlock>,
}
pub(crate) struct LanguageBlock {
    pub norm1: Tensor,
    pub qkv: Linear,
    pub proj: Linear,
    pub norm2: Tensor,
    pub gate_up: Linear,
    pub down: Linear,
}
pub(crate) struct Weights {
    pub towers: Vec<Tower>,
    pub projector: Vec<Linear>,
    pub embed: Tensor,
    pub language: Vec<LanguageBlock>,
    pub norm: Tensor,
    pub head: Linear,
    pub token_values: Tensor,
    pub rope_frequencies: Tensor,
}
struct Loader<'a> {
    plan: Rc<kernels::gemm::Bf16LinearPlan>,
    map: HashMap<String, Tensor>,
    backend: &'a RuntimeBackend,
}
impl Loader<'_> {
    fn host(&mut self, key: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self
            .map
            .remove(key)
            .ok_or_else(|| Error::Other(format!("missing OpenVLA weight {key}")))?;
        if t.shape().dims() != shape {
            return Err(Error::Other(format!(
                "{key}: expected {shape:?}, got {:?}",
                t.shape().dims()
            )));
        }
        if t.dtype() == DType::BF16 {
            Ok(t)
        } else {
            Tensor::from_bf16(
                shape.to_vec(),
                &t.to_f32_vec()?
                    .into_iter()
                    .map(bf16::from_f32)
                    .collect::<Vec<_>>(),
            )
        }
    }
    fn tensor(&mut self, key: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self.host(key, shape)?;
        self.backend.to_device(&t)
    }
    fn linear(&mut self, key: &str, input: usize, output: usize, bias: bool) -> Result<Linear> {
        self.packed(&[key], input, &[output], bias)
    }
    fn packed(
        &mut self,
        keys: &[&str],
        input: usize,
        outputs: &[usize],
        bias: bool,
    ) -> Result<Linear> {
        let n: usize = outputs.iter().sum();
        let mut packed = vec![bf16::ZERO; input * n];
        let mut biases = Vec::new();
        let mut base = 0;
        for (&key, &out) in keys.iter().zip(outputs) {
            let t = self.host(&format!("{key}.weight"), &[out, input])?;
            let values = t.as_bf16()?;
            if bias {
                for i in 0..input {
                    for j in 0..out {
                        packed[i * n + base + j] = values[j * input + i];
                    }
                }
            } else {
                packed[base * input..(base + out) * input].copy_from_slice(values);
            }
            if bias {
                biases.extend_from_slice(self.host(&format!("{key}.bias"), &[out])?.as_bf16()?);
            }
            base += out;
        }
        Ok(Linear {
            plan: self.plan.clone(),
            weight: self.backend.to_device(&Tensor::from_bf16(
                if bias { vec![input, n] } else { vec![n, input] },
                &packed,
            )?)?,
            bias: if bias {
                Some(
                    self.backend
                        .to_device(&Tensor::from_bf16(vec![n], &biases)?)?,
                )
            } else {
                None
            },
        })
    }
    fn norm(&mut self, key: &str, width: usize) -> Result<Norm> {
        Ok(Norm {
            weight: self.tensor(&format!("{key}.weight"), &[width])?,
            bias: self.tensor(&format!("{key}.bias"), &[width])?,
        })
    }
    fn scale(&mut self, key: &str, width: usize, rows: usize) -> Result<Tensor> {
        let t = self.host(key, &[width])?;
        let vals = t.as_bf16()?.repeat(rows);
        self.backend
            .to_device(&Tensor::from_bf16(vec![rows, width], &vals)?)
    }
}
impl Weights {
    pub fn load(path: &Path, backend: &RuntimeBackend) -> Result<Self> {
        let (map, _) = apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
        let mut l = Loader {
            map,
            backend,
            plan: Rc::new(kernels::gemm::Bf16LinearPlan::new(backend.context())?),
        };
        let mut towers = Vec::new();
        for (name, width, mlp, depth, registers) in [
            ("featurizer", 1024, 4096, 23, true),
            ("fused_featurizer", 1152, 4304, 26, false),
        ] {
            let root = format!("vision_backbone.{name}");
            let conv = l.tensor(
                &format!("{root}.patch_embed.proj.weight"),
                &[width, 3, 14, 14],
            )?;
            let bias = l.tensor(&format!("{root}.patch_embed.proj.bias"), &[width])?;
            let position = l
                .tensor(&format!("{root}.pos_embed"), &[1, 256, width])?
                .reshape(vec![256, width])?;
            let prefix = if registers {
                let cls = l.host(&format!("{root}.cls_token"), &[1, 1, width])?;
                let reg = l.host(&format!("{root}.reg_token"), &[1, 4, width])?;
                let mut values = cls.as_bf16()?.to_vec();
                values.extend_from_slice(reg.as_bf16()?);
                Some(backend.to_device(&Tensor::from_bf16(vec![5, width], &values)?)?)
            } else {
                None
            };
            let mut blocks = Vec::new();
            for i in 0..depth {
                let p = format!("{root}.blocks.{i}");
                blocks.push(VisionBlock {
                    norm1: l.norm(&format!("{p}.norm1"), width)?,
                    qkv: l.linear(&format!("{p}.attn.qkv"), width, 3 * width, true)?,
                    proj: l.linear(&format!("{p}.attn.proj"), width, width, true)?,
                    norm2: l.norm(&format!("{p}.norm2"), width)?,
                    fc1: l.linear(&format!("{p}.mlp.fc1"), width, mlp, true)?,
                    fc2: l.linear(&format!("{p}.mlp.fc2"), mlp, width, true)?,
                    scale1: if registers {
                        Some(l.scale(&format!("{p}.ls1.scale_factor"), width, 261)?)
                    } else {
                        None
                    },
                    scale2: if registers {
                        Some(l.scale(&format!("{p}.ls2.scale_factor"), width, 261)?)
                    } else {
                        None
                    },
                });
            }
            let plan = kernels::convolution::Conv2dPlan::prepare(
                backend.context(),
                kernels::convolution::Conv2dShape {
                    batch: 1,
                    channels: 3,
                    height: 224,
                    width: 224,
                    output_channels: width,
                    kernel: 14,
                    stride: 14,
                },
            )?;
            towers.push(Tower {
                width,
                prefix,
                position,
                conv,
                bias,
                plan,
                blocks,
            });
        }
        let projector = vec![
            l.linear("projector.fc1", 2176, 8704, true)?,
            l.linear("projector.fc2", 8704, 4096, true)?,
            l.linear("projector.fc3", 4096, 4096, true)?,
        ];
        let embed = l.tensor("language_model.model.embed_tokens.weight", &[32064, 4096])?;
        let mut language = Vec::new();
        for i in 0..32 {
            let p = format!("language_model.model.layers.{i}");
            language.push(LanguageBlock {
                norm1: l.tensor(&format!("{p}.input_layernorm.weight"), &[4096])?,
                qkv: l.packed(
                    &[
                        &format!("{p}.self_attn.q_proj"),
                        &format!("{p}.self_attn.k_proj"),
                        &format!("{p}.self_attn.v_proj"),
                    ],
                    4096,
                    &[4096; 3],
                    false,
                )?,
                proj: l.linear(&format!("{p}.self_attn.o_proj"), 4096, 4096, false)?,
                norm2: l.tensor(&format!("{p}.post_attention_layernorm.weight"), &[4096])?,
                gate_up: l.packed(
                    &[&format!("{p}.mlp.gate_proj"), &format!("{p}.mlp.up_proj")],
                    4096,
                    &[11008; 2],
                    false,
                )?,
                down: l.linear(&format!("{p}.mlp.down_proj"), 11008, 4096, false)?,
            });
        }
        let norm = l.tensor("language_model.model.norm.weight", &[4096])?;
        let head = l.linear("language_model.lm_head", 4096, 32064, false)?;
        let token_values = backend.to_device(&Tensor::from_f32(
            vec![32064, 1],
            &(0..32064).map(|x| x as f32).collect::<Vec<_>>(),
        )?)?;
        // Match the reference's CPU FP32 pow then reciprocal initialization.
        // Recomputing powf on CUDA changes four frequencies by one ULP.
        let frequencies = (0..64)
            .map(|i| (10000_f64.powf(i as f64 / 64.0) as f32).recip())
            .collect::<Vec<_>>();
        let rope_frequencies = backend.to_device(&Tensor::from_f32(vec![64], &frequencies)?)?;
        Ok(Self {
            towers,
            projector,
            embed,
            language,
            norm,
            head,
            token_values,
            rope_frequencies,
        })
    }
}
