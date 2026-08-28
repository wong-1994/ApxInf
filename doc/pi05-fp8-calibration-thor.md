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
- Validator revision: `4f6f95dd2752b790137609440d69675eff3e4a4a`
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
  --validator-revision 4f6f95dd2752b790137609440d69675eff3e4a4a \
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
| Mean absolute error | 0.00790789 |
| RMSE | 0.0131255 |
| Relative L2 | **0.0290504** |
| Relative-L2 gate | **PASS** (≤ 0.20) |

The raw deployed BF16/FP8 actions and aggregate calculations are stored in the
generated `evidence/thor-validation-text-only-loaded.json` on the validation host. Its
SHA-256 is
`a933818e2ae7abdec899ae6c22769dfc2f87964bc4be3e36e0f79f4869a124f4`.

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

Latency sampling uses identical images, the README's T=10 text-only prompt,
explicit noise, H=10, 10 warmups, 30 measured calls, and the model subspan of
`Pi05Policy.infer`. Returning host action arrays includes the synchronizing
device-to-host copy. The collector is disabled for both inference runs.

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
  --validator-revision 4f6f95dd2752b790137609440d69675eff3e4a4a \
  --out evidence/thor-h10-t10-regression-text-only.json
```

Two aligned runs exposed a shared-host limitation:

| Run | Precision | P50 | P95 | Mean | Host state discovered afterward |
|---|---|---:|---:|---:|---|
| A | BF16 | 73.77 ms | 74.23 ms | 73.79 ms | low contention, but the process filter was incomplete |
| A | calibrated FP8 | 42.99 ms | 50.64 ms | 43.82 ms | low contention, but the process filter was incomplete |
| B | BF16 | 79.76 ms | 80.59 ms | 79.86 ms | concurrent CUDA compilation |
| B | calibrated FP8 | 63.01 ms | 65.36 ms | 63.43 ms | concurrent CUDA compilation |

Run A is close to the README's pre-instrumentation baselines (72.45 ms BF16,
41.16 ms FP8), but a later global process audit showed that checking one known
build path was insufficient on this 17-user host. Run B's committed-validator
artifact was definitely contaminated by several other users' `nvcc`/`cicc`
processes and shared power/clock effects. A non-invasive wait for all CUDA
compilers to reach zero never found an idle window. Therefore this report does
**not** claim a production steady-state latency or prove the no-regression gate;
those two performance requirements remain pending an exclusive or verified-idle
Thor window with fixed clocks.

Raw model latency samples in milliseconds:

```text
BF16: [73.514, 73.154, 73.894, 73.661, 73.465, 73.533, 73.709, 73.881, 73.748, 73.458, 73.835, 73.429, 73.472, 73.954, 73.904, 74.231, 73.666, 73.859, 74.225, 74.291, 73.485, 73.527, 73.431, 73.768, 74.027, 74.855, 73.917, 73.820, 73.911, 74.017]
FP8:  [50.636, 50.941, 51.436, 42.711, 42.914, 43.565, 43.015, 42.831, 42.988, 42.954, 42.829, 43.235, 42.995, 42.880, 43.088, 42.969, 43.260, 43.120, 43.465, 43.001, 42.919, 42.874, 42.916, 43.085, 43.109, 43.038, 42.834, 42.912, 42.883, 43.256]
```

Run B's complete raw regression artifact is
`evidence/thor-h10-t10-regression-text-only.json` on the validation host,
SHA-256
`28b0cc6690f5b47c6752e9a49283000d27b0a72bfcd1bd85ed8db5b0316b12d2`.
Run A's raw samples are preserved above even though its JSON was superseded by
the provenance rerun.

## Remaining limitations

- This report's H=50 accuracy result is an offline action comparison over 16
  observations. The separate deployment-matched H=10 sample sweep and complete
  500-episode LIBERO-10 campaign are documented in
  [`pi05-fp8-calibration-libero10.md`](pi05-fp8-calibration-libero10.md).
- The tested relative-L2 threshold is an explicit gate for this fixture, not a
  framework-wide precision contract.
- This report's H=50 sample-count experiment shows that 8 is insufficient but
  does not prove convergence at 16. The H=10 study finds that 10 balanced
  Observations reproduce baseline LIBERO-10 task accuracy even though per-site
  absmax values remain unconverged at 640 samples.
- Formal steady-state and instrumentation-regression conclusions require a
  verified-idle Thor window; shared compilation made repeated timing unstable.
- Calibration profiles and tactic databases remain separate. This evidence does
  not retune or validate every alternate tactic database.
