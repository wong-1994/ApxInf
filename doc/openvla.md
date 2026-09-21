# OpenVLA

The `openvla` / `openvla_cuda` registrations load the original fused
DINOv2 + SigLIP, Llama2-7B OpenVLA architecture in BF16, including compatible
merged fine-tuned weights. Validated checkpoints include the original
`openvla-7b` and official `openvla-7b-finetuned-libero-spatial`. Other architectures,
quantized weights, CPU execution, and additional camera/state inputs are not
supported by this implementation.

## Public policy

```python
from PIL import Image
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained(
    "/path/to/openvla-7b", device="cuda:0", precision="bf16",
    unnorm_key="bridge_orig",
)
result = policy.infer({
    "image": Image.open("observation.png").convert("RGB"),
    "prompt": "put spoon on towel",
})
print(result["actions"])  # float32 [1, 7], checkpoint action units
print(result["token_ids"])
```

`OpenVlaPolicy.from_pretrained(..., model_runner=runner)` supports injection
through the existing native runner interface. The maintained policy uses
Pillow, NumPy and the native `HfTokenizer`; it does not import PyTorch or execute checkpoint
Python code. It applies the checkpoint's resize/normalization parameters and
prompt template, then decodes action bins and checkpoint quantiles on the host.
These are public input and output boundaries.

## Native contract and ownership

The Rust input is preprocessed host F32/BF16 pixels `[6, 50176]` (DINO channels
then SigLIP channels) and token IDs including BOS `1` and suffix `29871`.
The output is a CUDA F32 `[1, 7]` tensor of integral token IDs. The policy turns
these into continuous actions, including the reference's EOS handling.
The runtime requires at least two prompt tokens and reserves room for 256 image
tokens and seven action steps within a 2048-token context.

- `openvla/config.rs` and `load.rs` validate the checkpoint and construct the family.
- `openvla/weights.rs` owns load-time packing, native convolution plans and a
  prepared cuBLAS linear plan.
- `openvla/model/` performs both vision towers, projector, prefill and seven
  greedy decode steps entirely on CUDA. Intermediate token IDs and KV stay on GPU.
- `openvla/model_runner.rs` binds public inputs, owns graph/workspace lifetimes
  and implements `VlaRuntime` preparation.

Default inference requires a whole-model CUDA Graph. Explicit `Eager`,
`PreferGraph` and `RequireGraph` preparation policies are supported. A plan is
specific to its input profile and tuning-session identity/generation. Replacing
or changing tuning invalidates it; clearing the runner's cache does not destroy
an independently retained plan. Input values are rebound before each run.
Returned device output aliases plan workspace and must be consumed before that
plan runs again. The Python policy returns an independent host array.

## Numerical execution profile

The reference profile is `openvla/openvla-7b` revision
`47a0ec7fc4ec123775a391911046cf33cf9ed83f`, original OpenVLA source
`c8f03f48af692657d3060c19588038c7220e9af9`, BF16, `bridge_orig`.
The reference uses Transformers 4.40.1, timm 0.9.10 and PyTorch 2.9.1 + CUDA 13.0.
Native validation targets NVIDIA Thor (SM110). Language attention in the
reference is explicitly `eager` (`LlamaAttention`). Verify the instantiated
module type: with a pre-created config, Transformers 4.40.1 can auto-select
SDPA despite an `attn_implementation="eager"` keyword unless the config is
explicitly set. SDPA is a different numerical reference profile.

Numerical choices are explicit: cuDNN BF16 convolution, Welford vision
LayerNorm, FA2 head-64/head-72 instantiations with separate template symbols
and unfused score scaling, 1 MiB cuBLASLt bias-epilogue heuristic workspace,
8 MiB prepared cuBLAS language workspace, materialized BF16 causal attention,
rounded RMSNorm/RoPE boundaries, and first-index argmax tie handling.
RoPE inverse frequencies are prepared once in `weights.rs` with the reference's
FP32 power/reciprocal boundary and retained on GPU. Recomputing frequencies with
CUDA `powf` changes four values by one ULP; near a full rotation this can change
BF16 queries and eventually action tokens.
CUDA 13.2 and 13.0 compile different FP32 `erf` results near GELU's negative
tail; matching the reference therefore requires the CUDA 13.0 build profile.
Set `CUDA_PATH` to that toolkit and `CUDNN_LIB_DIR` to a directory containing
`libcudnn.so.9`; matching runtime libraries must be on the loader path.

The precision gate requires identical action tokens and continuous actions at
`atol=rtol=1e-2`. Validation includes three synthetic diagnostic images and
1,000 real BridgeData V2 frames spanning 83 trajectories and 75 instructions;
all 7,000 action tokens match the reference. This measures runtime equivalence
on fixed observations; closed-loop task success is a separate evaluation.
GPU graph and public-policy acceptance evidence is
kept privately under the worktree's `devlocal/openvla-port/`; performance is a
separate best-effort gate.

For LIBERO-Spatial, use checkpoint revision
`962318cec55ac10993ff0f5f43eda9a270b4c873` with `unnorm_key="libero_spatial"`.
The simulator adapter must reproduce the official camera rotation, JPEG and
Lanczos3 resize, 0.9-area center crop, and gripper conversion. These benchmark
transforms happen outside `OpenVlaPolicy`; the policy itself retains the
checkpoint processor contract. Do not reuse `bridge_orig` normalization for
this checkpoint.

`OpenVlaPolicy` returns continuous actions as float32. For paired closed-loop
evaluation, cast reference actions to float32 at the same control boundary
before gripper conversion. Float64 versus float32 controls can produce different
simulator trajectories even when action tokens are identical.

A paired LIBERO-Spatial smoke run covers ten tasks with initial state index 0:
both runtimes succeed on eight tasks, with identical tokens and control actions
across 1,300 steps. One independent simulator image differs slightly; replaying
both images through both runtimes preserves exact token agreement. This small
smoke run is not the official 500-trial benchmark.
