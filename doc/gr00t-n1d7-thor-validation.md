# GR00T N1.7 Thor validation

## Provenance

- Reference: NVIDIA `Isaac-GR00T`, revision
  `9c7e746b2cd37a810070a98ef41d290a07e806c2`.
- Checkpoint: `nvidia/GR00T-N1.7-3B`, snapshot
  `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- Target: NVIDIA Thor, SM110, CUDA 13.0, BF16.
- Official full-path fixture SHA256:
  `3f6924d005b18ad61d6767a51764115df3225b0494a6fca05f1524245590d092`.
- Official action-head fixture SHA256:
  `ba966f07f07c8e7cfd2e916e841700cdf275f27b873ffd83b7807e5125ad79e7`.

No other branch, previous GR00T port, or experimental port commit was used.

## Correctness and public path

The fixture fixes processor output, state, embodiment 0, and initial BF16 noise.
The Rust runtime executes the Qwen3-VL backbone and four Euler steps at timestep
buckets `[0, 250, 500, 750]`. Against the official NVIDIA BF16 output:

- shape: `[40, 132]`;
- max absolute error: `0.0625`;
- mean absolute error: `0.0071737389`;
- relative L2 error: `0.010037194`;
- three repeated runs: `max_abs=0` between runs.

`AutoPolicy.from_pretrained` resolved `Gr00tN1d7` to `GrootPolicy`, entered the
PyO3 `infer_groot` binding, and reproduced the same metrics. The standalone
runner is `scripts/validate_gr00t_n1d7.py`.

## Thor performance and optimization debt

The correctness build measured full backbone-to-action inference after one
warmup at `1796.84`, `1791.48`, and `1786.26` ms. These are debug-build
measurements and are not release performance claims.

Implemented optimization work:

- a device-only BF16 cross-SDPA kernel supports independent Q/KV lengths and
  head dimensions 48 and 64;
- Qwen text KV storage is allocated once and reset between requests;
- action weights and the backbone stay resident on the device;
- exact caller noise is quantized once at the PyO3 boundary and uploaded once.

Remaining measured/audited debt:

- `PreparedInference` is eager; CUDA Graph capture is not enabled. Host row
  filtering, AdaLayerNorm scaffolds, ReLU, category selection, and tail slicing
  allocate or synchronize, so capture would currently be invalid.
- Category-specific matrices are selected from host banks and uploaded per
  request instead of being cached per embodiment.
- Action AdaLayerNorm still round-trips its activation and conditioning through
  the host in every DiT block. This is the dominant action-head graph blocker.
- The maintained Qwen3-VL vision path still has host-backed slicing/reshape,
  positional interpolation, scatter-add, and merge scaffolds.
- Action intermediates use dynamic tensor allocation rather than a prepared
  static workspace. KV is static, but the vision/action temporary buffers are
  not.

The optimization status is therefore **best effort with performance debt**:
functional and deterministic BF16 parity passes, cross-attention stays on the
GPU, but CUDA Graph and fully static workspace claims are intentionally not
made.
