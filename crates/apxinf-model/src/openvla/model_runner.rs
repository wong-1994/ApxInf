//! Preparation, input transfers and fixed-profile CUDA Graph ownership.
use super::{
    backend::{kernels, DeviceBuffer, RuntimeBackend},
    model::Model,
};
use crate::{
    accelerator::cuda::{transfers, tuning},
    vla::*,
};
use apxinf_core::{Backend, DType, Device, Error, Graph, Result, Tensor};
use half::bf16;
use std::{cell::RefCell, rc::Rc, sync::Arc};
pub struct OpenVlaModelRunner {
    backend: Arc<RuntimeBackend>,
    model: Rc<Model>,
    cached: RefCell<Option<Rc<Plan>>>,
}
struct Plan {
    spec: InferenceSpec,
    backend: Arc<RuntimeBackend>,
    model: Rc<Model>,
    pixels: Tensor,
    ids: DeviceBuffer,
    workspace: kernels::GraphWorkspace,
    graph: Option<Box<dyn Graph>>,
    output: Option<Tensor>,
    fallback: Option<String>,
    session: Arc<tuning::TuningSession>,
    generation: u64,
    running: RefCell<()>,
}
struct Prepared(Rc<Plan>);
impl Plan {
    fn current(&self) -> bool {
        let t = self.backend.context().tuning();
        Arc::ptr_eq(&t, &self.session) && t.generation() == self.generation
    }
    fn bind(&self, request: &VlaRequest<'_>) -> Result<()> {
        let o = request.observation;
        if !self.spec.matches(o)
            || o.token_ids.iter().any(|&x| x >= 32064)
            || o.token_ids.first() != Some(&1)
            || o.token_ids.last() != Some(&29871)
        {
            return Err(Error::Other(
                "OpenVLA input profile or token IDs invalid (BOS=1, suffix=29871 required)".into(),
            ));
        }
        if o.state.is_some()
            || o.action_mask.is_some()
            || request.metadata.attention_mask.is_some()
            || request.metadata.planning.is_some()
            || request.metadata.image_grid_thw.is_some()
            || request.metadata.embodiment_id.is_some()
        {
            return Err(Error::Other(
                "OpenVLA does not accept state, masks or planning metadata".into(),
            ));
        }
        if !matches!(request.initial_latent, InitialLatent::Generate { .. }) {
            return Err(Error::Other(
                "OpenVLA has no continuous initial latent".into(),
            ));
        }
        let VisionObservation::Patches(t) = &o.vision else {
            return Err(Error::Other("OpenVLA expects preprocessed pixels".into()));
        };
        if t.shape().dims() != [6, 50176]
            || t.device() != Device::Cpu
            || !matches!(t.dtype(), DType::F32 | DType::BF16)
        {
            return Err(Error::Other(
                "OpenVLA expects host F32/BF16 pixels [6,50176]".into(),
            ));
        }
        let host = if t.dtype() == DType::BF16 {
            t.clone()
        } else {
            Tensor::from_bf16(
                vec![6, 50176],
                &t.as_f32()?
                    .iter()
                    .map(|&v| bf16::from_f32(v))
                    .collect::<Vec<_>>(),
            )?
        };
        self.backend.synchronize()?;
        transfers::copy_cpu_to_cuda(&host, &self.pixels)?;
        self.ids
            .copy_from_host(
                &o.token_ids
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .map_err(Error::Cuda)
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let _guard = self
            .running
            .try_borrow_mut()
            .map_err(|_| Error::Other("OpenVLA plan is already running".into()))?;
        if !self.current() {
            return Err(Error::Other(
                "OpenVLA prepared plan invalidated by tuning changes".into(),
            ));
        }
        self.bind(request)?;
        tuning::without_autotune(|| {
            let out = if let Some(graph) = &self.graph {
                graph.replay()?;
                self.output.as_ref().unwrap().clone()
            } else {
                kernels::with_workspace(&self.workspace, || {
                    self.model.forward(
                        self.backend.context(),
                        &self.pixels,
                        &self.ids,
                        self.spec.token_count,
                    )
                })?
            };
            Ok(Action::new(out))
        })
    }
}
impl PreparedInference for Prepared {
    fn spec(&self) -> &InferenceSpec {
        &self.0.spec
    }
    fn status(&self) -> PreparationStatus {
        if !self.0.current() {
            PreparationStatus::Invalidated
        } else {
            PreparationStatus::Ready {
                mode: if self.0.graph.is_some() {
                    ExecutionMode::Graph
                } else {
                    ExecutionMode::Eager
                },
                fallback_reason: self.0.fallback.clone(),
            }
        }
    }
    fn run(&self, r: &VlaRequest<'_>) -> Result<Action> {
        self.0.run(r)
    }
}
impl OpenVlaModelRunner {
    pub(crate) fn new(backend: Arc<RuntimeBackend>, model: Model) -> Self {
        Self {
            backend,
            model: Rc::new(model),
            cached: RefCell::new(None),
        }
    }
    fn plan(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
        sample: Option<&VlaRequest<'_>>,
    ) -> Result<Rc<Plan>> {
        spec.validate()?;
        if spec.image_layout.is_some() {
            return Err(Error::Other("OpenVLA requires preprocessed pixels".into()));
        }
        let workspace = kernels::GraphWorkspace::new(
            self.model.workspace_bytes(spec.token_count)?,
            self.backend.context().device_id(),
        )?;
        let pixels = self.backend.to_device(&Tensor::from_bf16(
            vec![6, 50176],
            &vec![bf16::ZERO; 6 * 50176],
        )?)?;
        let ids =
            DeviceBuffer::alloc_zeros(spec.token_count * 4, self.backend.context().device_id())
                .map_err(Error::Cuda)?;
        let t = self.backend.context().tuning();
        let mut plan = Plan {
            spec: *spec,
            backend: self.backend.clone(),
            model: self.model.clone(),
            pixels,
            ids,
            workspace,
            graph: None,
            output: None,
            fallback: None,
            session: t.clone(),
            generation: t.generation(),
            running: RefCell::new(()),
        };
        if let Some(r) = sample {
            plan.bind(r)?;
        }
        let run = || {
            plan.model.forward(
                plan.backend.context(),
                &plan.pixels,
                &plan.ids,
                spec.token_count,
            )
        };
        // prepare_for may tune real observations; placeholder preparation never tunes.
        if sample.is_some() {
            kernels::prepare_with_workspace(&plan.workspace, run)?;
        } else {
            tuning::without_autotune(|| kernels::prepare_with_workspace(&plan.workspace, run))?;
        }
        plan.backend.synchronize()?;
        if policy != ExecutionPolicy::Eager {
            match tuning::without_autotune(|| {
                plan.backend
                    .capture_graph(|| kernels::with_workspace(&plan.workspace, run))
            }) {
                Ok((graph, output)) => {
                    plan.graph = Some(graph);
                    plan.output = Some(output);
                }
                Err(e) if policy == ExecutionPolicy::PreferGraph => {
                    plan.fallback = Some(e.to_string())
                }
                Err(e) => return Err(e),
            }
        }
        plan.session = plan.backend.context().tuning();
        plan.generation = plan.session.generation();
        Ok(Rc::new(plan))
    }
}
impl VlaRuntime for OpenVlaModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [1, 7],
            patch_shape: [6, 50176],
            max_token_len: self.model.config.max_tokens,
            num_views: 1,
            image_size: 224,
            patch_size: 14,
            accepts_rgb_u8: false,
        }
    }
    fn infer(&self, r: &VlaRequest<'_>) -> Result<Action> {
        let spec = r.observation.inference_spec();
        let reuse = self
            .cached
            .borrow()
            .as_ref()
            .filter(|p| p.spec == spec && p.current())
            .cloned();
        let plan = match reuse {
            Some(p) => p,
            None => {
                let p = self.plan(&spec, ExecutionPolicy::RequireGraph, None)?;
                *self.cached.borrow_mut() = Some(p.clone());
                p
            }
        };
        plan.run(r)
    }
    fn prepare(&self, s: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(s, ExecutionPolicy::RequireGraph)
    }
    fn prepare_with_policy(
        &self,
        s: &InferenceSpec,
        p: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(Prepared(self.plan(s, p, None)?)))
    }
    fn prepare_for(
        &self,
        r: &VlaRequest<'_>,
        p: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(Prepared(self.plan(
            &r.observation.inference_spec(),
            p,
            Some(r),
        )?)))
    }
    fn clear_prepared(&self) -> Result<()> {
        *self.cached.borrow_mut() = None;
        Ok(())
    }
    fn infer_host_f32(&self, r: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.backend.to_cpu(self.infer(r)?.tensor())?.to_f32_vec()
    }
    fn execution_mode(&self) -> &'static str {
        "graph"
    }
    fn action_token_shape(&self) -> Option<[usize; 2]> {
        Some([1, 7])
    }
    fn infer_action_tokens(&self, r: &VlaRequest<'_>, stop: Option<u32>) -> Result<Tensor> {
        if stop.is_some() {
            return Err(Error::Other(
                "OpenVLA returns seven device-selected candidates; policy handles EOS".into(),
            ));
        }
        Ok(self.infer(r)?.into_tensor())
    }
}
