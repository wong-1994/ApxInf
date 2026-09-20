# Qwen-Drive planning runtime

Qwen-Drive is registered as a CUDA VLA runtime. The supported workload is
trajectory planning, with either a direct prompt or internal reasoning before
planning. VQA/text generation and BEV perception are not public or development
entry points. The Qwen vision/text backbone provides scene context to the planner;
it is not exposed as a separate Qwen-Drive language model.

## Ownership

The family follows the [model layer ownership rules](model-layer-architecture.md):

- `load.rs` validates the variant/assets, loads both checkpoints, chooses the
  physical projection layout once, and constructs a `QwenDriveModelRunner`.
- `weights/host.rs` maps checkpoint names; `weights/bf16.rs` owns device weight
  representations. Weights do not depend on the model or runner.
- `model/mod.rs` connects the multimodal prefix, optional reasoning and flow
  solver. Reasoning tokens are an intermediate result of planning.
- `model/blocks/bf16.rs` is the single BF16 network implementation. Vision,
  hybrid text blocks and planner computations are grouped internally in this
  file. There is no directory per modality or speculative variant dispatch.
- `model_runner/runner.rs` implements `VlaRuntime`, validates canonical requests,
  binds inputs/RNG and owns mutable state. Vision position interpolation has a
  bounded cache retained by the runner across requests. KV/recurrent state is
  replaced per request, after dropping graphs that reference its buffers.
- `model_runner/prepare.rs` owns the optional per-layer GDN graph resources.
  Blocks submit an eager operation through an execution callback; the runner
  does not select or access concrete BF16 Blocks.

Run `python3 scripts/check_qwen_drive_module_boundaries.py` to check the
explicit dependency directions.

## Loading and requests

Use `AutoPolicy.from_pretrained(checkpoint, mode="direct_planning")`, or select
`mode="reasoning_planning"`. Both use the shared Python `ModelRunner.load`
binding, which loads native `LoadedModel::Vla`. `model_variant="auto"` and
`"bf16"` are supported. The planner checkpoint is required: pass
`planner=...` to the policy (native named asset `assets["planner"]`) or
provide `planner-sft` under the model directory. A missing planner is an error;
there is no backbone-only fallback.

The policy owns prompt construction, patchification, conditioning normalization
and trajectory decoding. Native requests contain canonical patches, image grids,
prompt token IDs and a host F32 conditioning vector, packed in this order:
normalized history poses, history velocity, history acceleration, ego status,
navigation command. Section sizes come from the model configuration.

`PlanningOptions` carries the flow step count and optional reasoning token bounds
and turn delimiters. The policy's `num_steps` reaches the actual solver. Provided
noise is preserved, validated against the configured horizon/dimension and copied
into a private solver buffer. The CUDA VLA API can alternatively generate noise
from the request RNG. Public actions use the configured `[horizon, 3]` shape;
the Python policy includes the batch dimension and decodes physical coordinates.

The family uses common `tactics`/`autotune` loading options. The former standalone
`APXINF_QWEN_TUNING_DIR`/`APXINF_QWEN_AUTOTUNE` loader controls are superseded by
those shared options. `APXINF_QWEN_LINEAR_TUNED` still selects the projection
layout at load time; changing it after loading cannot change weight layout.

## Execution and validation limits

Existing BF16 kernels and fused operations are retained. This organization change
does not establish a new fusion scheme or a performance baseline. Direct planning
skips the unused final language projection; reasoning still computes token logits.

`APXINF_QWEN_DECODE_GRAPH` enables the inherited local GDN decode graph path,
including separate convolution parity captures. It does not capture vision,
full-attention cache management or the planner/flow loop. Full VLA `prepare`
returns an explicit unsupported error; callers must not treat local captures as
a prepared full-model execution plan. Vision position interpolation still uses
host computation on cache misses.

Local checks cover Rust types, portable conditioning/configuration logic and
Python direct/reasoning request wiring. A CUDA-feature check on a machine without
a CUDA toolkit does not compile device kernels or run the model. Before claiming
numerical equivalence or performance, run both planning modes on target GPUs
with the same checkpoint, patches, token IDs, conditioning and exact supplied
noise as the integrated baseline. Compare trajectories and reasoning tokens,
exercise multiple flow step counts, and test the GDN graph toggle with enough
reasoning tokens to enter replay. Record warm-up, steady-state latency and peak
memory separately. Both datacenter and edge targets need their own evidence.

### Biased planner projections

The planner's one-row MLPs and adaLN modulation use `gemm::bf16_addmv` with
checkpoint-layout weights. Its contract adds bias before the final BF16
rounding. On Thor, the BF16-output cuBLAS GEMV path can round the dot product
before adding bias. The operator therefore keeps the accumulator and bias in
FP32, using BF16 matrix/vector operands, then casts the completed result to
BF16. This preserves the model layout and uses workspace-backed device buffers.

CUDA regressions cover cancellation-sensitive projection inputs and graph
capture/replay with an updated bias. These operator checks do not establish
whole-model graph support or downstream planning accuracy; those remain subject
to the validation limits above.
