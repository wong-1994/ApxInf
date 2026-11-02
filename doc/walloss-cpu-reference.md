# WallOSS CPU reference

WallOSS supports `Device::Cpu` with `model_variant="fp32_reference"` (also
selected by `auto` on CPU). Its self-contained `cpu_walloss` implementation
executes unfused FP32 operations, without a CUDA toolkit or GPU. This is a
numerical reference path; it does not emulate BF16/FP8 rounding or promise
production CPU latency.

## Build and run

Linux and macOS use the same CPU backend with an installed multithreaded
OpenBLAS library:

```bash
cargo test -p apxinf-model --features openblas --test cpu_walloss
```

Without the `openblas` feature, GEMM uses the serial Rust fallback, which is
usable for the tiny contract fixture but very slow for the full checkpoint.
For OpenBLAS outside the system library search path, set `OPENBLAS_LIB_DIR` to
its library directory; runtime dynamic-library lookup must also be configured
appropriately.

```rust
use apxinf_core::Device;
use apxinf_model::{AutoModel, LoadOptions};

let model = AutoModel::load_model(
    Device::Cpu,
    "/path/to/wall-oss-0.5",
    &LoadOptions {
        model_name: Some("walloss".into()),
        model_variant: Some("fp32_reference".into()),
        ..LoadOptions::default()
    },
)?;
// Existing VlaRuntime requests and infer_host_f32 are unchanged.
```

`WallossPolicy.from_pretrained(..., device="cpu", model_variant="fp32_reference")`
uses the same native path when the Python extension is built for CPU. Image
resize, tokenization, state binning and action unnormalization remain policy
responsibilities. `walloss-cuda` continues to require CUDA.

## Contract and ownership

The `cpu_walloss/` port owns its implementation independently of `walloss/`
(CUDA). It does not import that family's configuration, weights or model code.
The CUDA directory is unchanged. Copies are intentional; independent reference
fixtures establish agreement between ports.

- `config.rs` owns checkpoint configuration, validation and the Qwen2-VL image
  processor normalization.
- `geometry.rs` owns the host vision layout (window/full attention offsets,
  patch order, reverse indices) and multimodal position IDs.
- `schedule.rs` owns the fixed solver timeline and timestep encoding.
- `weights.rs` owns checkpoint mapping and converts F32/F16/BF16 tensors to
  FP32 before transposing every linear weight into the BLAS-ready
  `[input, output]` layout. FP8 is rejected.
- `model.rs` owns the vision tower, language prefix KV, action expert, flow
  loop and forward order in `CpuWallossModel`.
- `runner.rs` owns `CpuWallossRunner`: request validation, RGB patchification,
  position IDs, noise generation and explicit eager plans.
- `apxinf-cpu` owns generic FP32 math. This change adds masked GQA/MHA
  (`CpuAttentionMask::Causal`) and the multimodal/axial RoPE implementations
  behind the existing `Backend::rope_mrope` and `Backend::rope_vision_2d`
  contracts. Heads and views are scheduled serially; BLAS owns threading.

Inputs are canonical patch rows (merged-group order) or already resized RGB
`u8` in NHWC/NCHW. Patches are `[views * grid_h * grid_w, 3 * temporal_patch_size
* patch_size^2]`; RGB sizes are `grid_h * patch_size` by `grid_w * patch_size`
per view. The checkpoint's prompt must contain one image-token run per grid;
state stays encoded in the prompt. `image_grid_thw` selects non-default shapes.
All supplied views/tokens must be valid; padding masks and extra planning or
embodiment metadata are rejected.

`Eager` preparation runs the actual computation before reporting ready.
`PreferGraph` returns an eager plan with an explicit fallback reason.
`RequireGraph` fails. No CPU no-op graph is replayed. Explicit plans retain the
model after their original runner is dropped. Only unquantized checkpoint
weights and `auto`/`fp32_reference` are supported; CUDA calibration/tuning
options are rejected.

## Verification

```bash
cargo test -p apxinf-cpu --lib
cargo test -p apxinf-model --lib cpu_walloss::
cargo test -p apxinf-model --test cpu_walloss
cargo test -p apxinf-model --test cpu_walloss --features openblas
```

The default tests use a tiny synthetic checkpoint and cover public loading,
RGB/patch equivalence, changed RNG inputs, eager readiness, graph rejection,
invalid requests, vision geometry, multimodal positions, both RoPE layouts and
scheduler values.

The full-checkpoint comparison is opt-in and requires CUDA:

```sh
APXINF_WALLOSS_TEST_MODEL=/path/to/wall-oss-0.5 \
  cargo test --release -p apxinf-model --features cuda,openblas \
  --test walloss_cpu_reference -- --ignored --test-threads=1
```

It loads the same checkpoint and the same RGB/token/provided-latent request on
CPU FP32 and CUDA BF16, drops the CPU model before loading the GPU model, and
reports full-array cosine, relative L2 and maximum absolute error. The
comparison measures precision loss; it is not a robot task-success evaluation
and does not claim CPU production latency.

Result (2026-11-02, thor-3, `wall_oss_05_libero`, default 2x252 profile,
OpenBLAS 0.3.26 pthread, `OPENBLAS_NUM_THREADS=8`):

| Comparison | Cosine | Relative L2 | Max absolute |
| --- | ---: | ---: | ---: |
| CPU FP32 vs CUDA BF16 | 0.999338 | 0.03646 | 0.03681 |

The CUDA path used here is the one previously compared with the wall-x/PyTorch
reference (`walloss.md`, 2x252 BF16 cosine 0.999753), so this result places the
CPU port inside the same BF16 precision band rather than defining a new one.
