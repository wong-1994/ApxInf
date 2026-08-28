# PI0.5 static-FP8 calibration validation on Thor

Date: 2026-08-28

This report validates ApxInf's native Observation-based PI0.5 calibration path.
Calibration and inference use ApxInf only; FlashRT is not installed or imported,
and tactic selection remains a separate artifact and workflow.

## Environment and inputs

- Host: NVIDIA Thor, compute capability 11.0 (SM110), 20 SMs
- OS: Linux 6.8.12-tegra, aarch64, glibc 2.39
- Driver: 580.00
- CUDA toolkit: 13.0, nvcc 13.0.48
- Rust: 1.98.0
- Python extension: CPython 3.10 wheel built with maturin 1.15.0
- Calibration implementation revision:
  `4d38f1f1c6d909d635aa22b81dca032d2b0028f9`
- Validator revision: `5ace7ce` (the follow-up review commit changes only
  provenance fields and documentation)
- Checkpoint: `/home/wwxq/Projects/models/pi05_libero_base`
- Data: 16 deterministic records drawn across 14 tasks from the first-party
  LeRobot LIBERO v3 dataset (273,465 frames, 40 tasks). Each public Observation
  contains two 256×256 RGB images, the task prompt, and raw 8-D state. The
  `franka_libero` production preset uses `discrete_state=False`, so the standard
  Policy path deliberately drops state and tokenizes prompt text only.

A clean transferred source snapshot built successfully with:

```bash
export CUDA_PATH=/usr/local/cuda
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --workspace --release --features cuda
```

The clean release build auto-detected SM110/CUTLASS SM110a and completed in
19m05s. The native CPython 3.10 wheel then built and passed an import smoke test:

```bash
maturin build --release --features cuda --auditwheel skip \
  -m crates/apxinf-py/Cargo.toml --interpreter .venv/bin/python
pip install --no-deps target/wheels/apxinf_py-0.1.0-cp310-cp310-linux_aarch64.whl
```

## Reproduction

The representative-data calibration command was run twice with the same 16
`--input` Observation NPZs. Bash expands the inputs deterministically:

```bash
inputs=()
for sample in evidence/observations/sample-*.npz; do
  inputs+=(--input "$sample")
done
python scripts/calibrate_pi05.py \
  --model-dir /home/wwxq/Projects/models/pi05_libero_base \
  "${inputs[@]}" \
  --output evidence/calibration-text-run-1.json \
  --data-id dataset:lerobot-libero-v3-stratified16-text-only \
  --source-revision 4d38f1f1c6d909d635aa22b81dca032d2b0028f9 \
  --action-horizon 50 --margin 1.1 --seed 0
```

The second run changes only `--output`. Both files have SHA-256
`d07894c325eae30ff0806fb1da4c9eec2f729688e003fc6a938122d638252375`.
Their canonical content is also identical after excluding the explicitly
permitted `device` field. The profile uses E4M3FN, `absmax`, margin `1.1`, and
`max(amax*margin/448,1e-8)`.

Accuracy validation used two repeated profiles, the same 16 observations,
explicit sample-indexed noise, and a relative-L2 acceptance threshold fixed at
`0.20` before results were observed:

```bash
python scripts/validate_pi05_calibration.py \
  --model-dir /home/wwxq/Projects/models/pi05_libero_base \
  --profile evidence/calibration-text-run-1.json \
  --profile evidence/calibration-text-run-2.json \
  "${inputs[@]}" \
  --action-dim 7 --action-horizon 50 --seed 0 \
  --max-relative-l2 0.20 --warmup 1 --samples 1 \
  --validator-revision 5ace7ce \
  --out evidence/thor-validation-text-only-loaded.json
```

## Coverage, reproducibility, and accuracy

| Check | Result |
|---|---:|
| Required sites | 256 |
| Observed sites | 256 |
| Generated scales | 256 |
| Missing / unknown / unused sites | 0 / 0 / 0 |
| FP8 runtime accepted profile | yes |
| Equivalent calibration runs | yes |
| Compared deployed-action elements | 5,600 |
| Non-finite outputs | 0 |
| Maximum absolute error | 0.0624875 |
| Mean absolute error | 0.00792555 |
| RMSE | 0.0131236 |
| Relative L2 | **0.0290468** |
| Relative-L2 gate | **PASS** (≤ 0.20) |

The raw deployed BF16/FP8 actions and aggregate calculations are stored in the
generated `evidence/thor-validation-text-only-loaded.json` on the validation host. Its
SHA-256 is
`d7d9b4debaa11c263ae2bb1f9d3862a461dd2c0b7345f714ec25c5c43152883a`.

## Sample-count evidence and production recommendation

Prefix profiles were collected with the same ordering and seed policy:

| Samples | Sites below the 16-sample amax | Max relative amax delta | Mean relative amax delta |
|---:|---:|---:|---:|
| 4 | 151 / 256 | 53.92% | 6.96% |
| 8 | 103 / 256 | 49.41% | 3.48% |
| 16 | 0 / 256 | 0% | 0% |

Use `absmax`, margin `1.1`, base seed `0`, and the documented sample-index seed
sequence for this checkpoint/fixture. Treat 16 diverse observations as the
validated lower bound only. The 8→16 change is material, so this experiment
does not establish that 16 is statistically converged or sufficient for another
deployment. Production calibration data must be stratified over deployed tasks,
cameras, prompts, and states; increase its size until per-site maxima and
held-out business-action accuracy stabilize.

## Performance protocol and status

Formal latency sampling uses identical images, the README's T=10 text-only
prompt, explicit noise, H=10, 10 warmups, 30 measured calls, and the model
subspan of `Pi05Policy.infer`. Returning host action arrays includes the
synchronizing device-to-host copy. The collector is disabled for both inference
runs, and no other CUDA build or GPU process was active.

The exact performance command used the same profiles and a latency-only
Observation whose images came from `sample-000.npz` and whose prompt was the
README T=10 string `put both moka pots on the stove`:

```bash
python scripts/validate_pi05_calibration.py \
  --model-dir /home/wwxq/Projects/models/pi05_libero_base \
  --profile evidence/calibration-text-run-1.json \
  --profile evidence/calibration-text-run-2.json \
  --input evidence/latency-t10.npz \
  --action-dim 7 --action-horizon 10 --seed 0 \
  --max-relative-l2 0.20 --warmup 10 --samples 30 \
  --validator-revision 5ace7ce \
  --out evidence/thor-h10-t10-regression-text-only.json
```

| Precision | Min | P50 | P95 | Max | Mean | Stddev |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 73.15 ms | 73.77 ms | 74.23 ms | 74.85 ms | 73.79 ms | 0.33 ms |
| calibrated FP8 | 42.71 ms | 42.99 ms | 50.64 ms | 51.44 ms | 43.82 ms | 2.40 ms |

FP8 reduces P50 by 41.7% relative to BF16 for this aligned workload. Against
the README's pre-instrumentation collector-disabled baselines (72.45 ms BF16,
41.16 ms FP8), the new P50 values are +1.82% and +4.46%. Both are inside the 5%
material-regression boundary used for this report. That boundary was not supplied
by the ticket, so consumers needing a stricter release gate must rerun with their
own threshold.

Raw model latency samples in milliseconds:

```text
BF16: [73.514, 73.154, 73.894, 73.661, 73.465, 73.533, 73.709, 73.881, 73.748, 73.458, 73.835, 73.429, 73.472, 73.954, 73.904, 74.231, 73.666, 73.859, 74.225, 74.291, 73.485, 73.527, 73.431, 73.768, 74.027, 74.855, 73.917, 73.820, 73.911, 74.017]
FP8:  [50.636, 50.941, 51.436, 42.711, 42.914, 43.565, 43.015, 42.831, 42.988, 42.954, 42.829, 43.235, 42.995, 42.880, 43.088, 42.969, 43.260, 43.120, 43.465, 43.001, 42.919, 42.874, 42.916, 43.085, 43.109, 43.038, 42.834, 42.912, 42.883, 43.256]
```

The complete regression artifact is
`evidence/thor-h10-t10-regression-text-only.json` on the validation host,
SHA-256
`2f3a7fd4a5f606522136a9e9c8183013dcd9101f86f97eee0d5e719dcffa0e6b`.

## Remaining limitations

- Accuracy is an offline action comparison over 16 observations, not a complete
  LIBERO rollout-success campaign.
- The tested relative-L2 threshold is an explicit gate for this fixture, not a
  framework-wide precision contract.
- The sample-count experiment shows that 8 is insufficient but does not prove
  convergence at 16.
- Calibration profiles and tactic databases remain separate. This evidence does
  not retune or validate every alternate tactic database.
