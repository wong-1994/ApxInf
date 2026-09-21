#![cfg(feature = "cuda")]
//! Run with APXINF_OPENVLA_CHECKPOINT pointing at the original BF16 checkpoint.
use apxinf_core::{Backend, DType, Device, Error, Result, RngKey, Tensor};
use apxinf_cuda::{
    tuning::{TacticStore, TuningSession},
    CudaBackend,
};
use apxinf_model::{register_builtin_models, registry, vla::*, LoadOptions, ModelPrecision};
use std::sync::Arc;

#[test]
#[ignore = "requires Thor and the original OpenVLA-7B checkpoint"]
fn graph_rebind_lifetime_and_tuning_invalidation() -> Result<()> {
    let path = std::env::var("APXINF_OPENVLA_CHECKPOINT")
        .map_err(|_| Error::Other("set APXINF_OPENVLA_CHECKPOINT".into()))?;
    let backend = Arc::new(CudaBackend::new(0)?);
    register_builtin_models();
    let observation = Observation {
        vision: VisionObservation::Patches(Tensor::zeros(vec![6, 50176], DType::F32)),
        token_ids: vec![1, 29871],
        state: None,
        action_mask: None,
    };
    let request = VlaRequest::generated(&observation, RngKey::new(0, 0, 0));
    let retained;
    {
        let loaded = registry::get("openvla_cuda").unwrap()(
            std::path::Path::new(&path),
            Device::Cuda(0),
            backend.clone(),
            &LoadOptions {
                precision: ModelPrecision::Bf16,
                ..Default::default()
            },
        )?;
        let runtime = loaded.vla()?;
        let spec = observation.inference_spec();
        let eager = runtime.prepare_with_policy(&spec, ExecutionPolicy::Eager)?;
        retained = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        assert_eq!(
            retained.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                fallback_reason: None,
            }
        );
        let expected = backend
            .to_cpu(eager.run(&request)?.tensor())?
            .to_f32_vec()?;
        for _ in 0..2 {
            assert_eq!(
                backend
                    .to_cpu(retained.run(&request)?.tensor())?
                    .to_f32_vec()?,
                expected
            );
        }
        let mut changed = observation.clone();
        changed.token_ids = vec![1, 29871, 29871];
        assert!(retained
            .run(&VlaRequest::generated(&changed, RngKey::new(0, 0, 0)))
            .is_err());
        runtime.clear_prepared()?;
        assert!(retained.run(&request).is_ok());
    }
    // The plan owns model, native handles and arena even after its runner drops.
    assert!(retained.run(&request).is_ok());
    backend
        .context()
        .install_tuning(TuningSession::inference(TacticStore::default()))
        .map_err(Error::Cuda)?;
    assert_eq!(retained.status(), PreparationStatus::Invalidated);
    assert!(retained.run(&request).is_err());
    Ok(())
}
