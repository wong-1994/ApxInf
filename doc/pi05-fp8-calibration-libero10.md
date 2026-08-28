# PI0.5 FP8 calibration sample sweep on LIBERO-10

Date: 2026-08-28

This report measures how many representative Observations are needed for the
observation-driven static-FP8 calibration path to reproduce PI0.5 LIBERO-10
task accuracy. It uses the README deployment protocol rather than the H=50
offline protocol in `pi05-fp8-calibration-thor.md`.

## Protocol

- Hardware: NVIDIA Thor, compute capability 11.0 (SM110), 20 SMs
- Checkpoint: `/home/wwxq/Projects/models/pi05_libero_base`
- Calibration source revision:
  `4d38f1f1c6d909d635aa22b81dca032d2b0028f9`
- Precision: static FP8 E4M3FN, per-tensor absmax, margin 1.1
- Model input: two 224x224 RGB views and prompt text
- Discrete state: disabled, matching the README and `franka_libero` preset
- Action horizon: H=10
- Rollout: all 10 LIBERO-10 tasks, 50 trials per task, replan=5, seed 7

The source dataset is the first-party LeRobot LIBERO v3 dataset. Its first ten
task IDs are exactly the LIBERO-10 task set (the official benchmark presents
them in a different order). For every task:

- the first 8 demonstration episodes supplied the calibration pool;
- the next 4 episodes supplied the held-out pool;
- 8 uniformly spaced frames were selected from every episode.

This produced 640 calibration Observations from 80 episodes and 320 held-out
Observations from 40 disjoint episodes. Calibration inputs are ordered in
round-robin task order. Every tested prefix therefore has equal task coverage:

| Samples | Samples per task |
|---:|---:|
| 10 | 1 |
| 20 | 2 |
| 40 | 4 |
| 80 | 8 |
| 160 | 16 |
| 320 | 32 |
| 640 | 64 |

The split manifest contains the source task, episode, frame, prompt, and file
SHA256 for every Observation. Calibration and held-out sets have zero episode
or frame overlap.

## Scale convergence

All seven profiles contain the exact required 256-site set. Successive-prefix
absmax changes were:

| Prefixes | Changed sites | Mean relative delta | P95 | Maximum |
|---:|---:|---:|---:|---:|
| 10 -> 20 | 100 / 256 | 4.35% | 25.41% | 56.84% |
| 20 -> 40 | 91 / 256 | 3.64% | 24.69% | 37.38% |
| 40 -> 80 | 83 / 256 | 2.66% | 14.74% | 39.35% |
| 80 -> 160 | 99 / 256 | 3.48% | 20.35% | 41.95% |
| 160 -> 320 | 88 / 256 | 2.72% | 16.42% | 28.81% |
| 320 -> 640 | 86 / 256 | 2.08% | 10.67% | 49.74% |

The scales are not statistically converged at N=640. Absmax continues to find
rare larger activations, and increasing N does not monotonically improve FP8
resolution for the bulk of the distribution.

## Held-out action accuracy

BF16 and every FP8 profile used the same 320 held-out Observations and explicit
sample-indexed noise. Each row compares 22,400 final deployed action elements
(320 x H10 x 7), with no non-finite values.

| Calibration samples | Relative L2 | Mean abs | Maximum abs | Worst-task relative L2 |
|---:|---:|---:|---:|---:|
| 10 | **0.06682** | 0.00718 | 2.0240 | **0.14016** |
| 20 | 0.06739 | **0.00705** | 2.0631 | 0.14186 |
| 40 | 0.06716 | 0.00703 | 2.0484 | 0.14026 |
| 80 | 0.06737 | 0.00743 | 2.0640 | 0.14161 |
| 160 | 0.06821 | 0.00753 | 2.0562 | 0.14226 |
| 320 | 0.06844 | 0.00790 | **2.0279** | 0.14086 |
| 640 | 0.06863 | 0.00772 | 2.0465 | 0.14114 |

The N=10 profile has the best overall relative L2, while all profiles are on
the same action-error plateau. Larger representative sets discover more
absmax outliers but do not improve held-out business output.

## Formal LIBERO-10 rollout

The smallest balanced profile, N=10 (one Observation per task), was selected
before rollout because it had the best held-out relative L2. The exact README
in-process protocol completed all 500 unique `(suite, task, trial)` keys:

```bash
python3 scripts/eval_libero.py \
  --backend in-process \
  --model-dir /home/wwxq/Projects/models/pi05_libero_base \
  --precision fp8 \
  --calibration /path/to/profiles-h10/calibration-n10.json \
  --action-horizon 10 \
  --suite libero_10 --tasks all --trials-per-task 50 --seed 7 \
  --results-jsonl /path/to/results.jsonl \
  --summary-json /path/to/summary.json
```

| Task | Successes | Rate |
|---:|---:|---:|
| 0 | 47 / 50 | 94% |
| 1 | 50 / 50 | 100% |
| 2 | 50 / 50 | 100% |
| 3 | 49 / 50 | 98% |
| 4 | 50 / 50 | 100% |
| 5 | 50 / 50 | 100% |
| 6 | 46 / 50 | 92% |
| 7 | 49 / 50 | 98% |
| 8 | 30 / 50 | 60% |
| 9 | 44 / 50 | 88% |
| **Total** | **465 / 500** | **93.0%** |

All 500 ledger rows are unique and completed; there are no missing runs or
technical errors. The 95% Wilson interval is 90.42% to 94.92%.

For comparison, the README reports an official PI0.5 reference of 92.4%, Thor
BF16 at 464/500 (92.8%), and the historical Thor FP8 profile at 470/500
(94.0%). The N=10 observation-driven profile therefore reproduces the
deployment accuracy baseline: it is 0.2 percentage points above the published
Thor BF16 result and 0.6 points above the official reference, while trailing
the historical FP8 result by 1.0 point.

## Evidence and conclusion

Artifacts remain on the validation host under:

```text
/home/wwxq/Projects/apxinf-calibration-ticket03-codex-20260828/
  evidence/libero10-sample-sweep-v2/
```

Key SHA256 values:

| Artifact | SHA256 |
|---|---|
| Split manifest | `5a657250e0022a4fe9c5d38aeeba4676376fc63659576ecf3924e90e08f68c37` |
| H10 held-out sweep | `3622c0649750154efc2ec4194d7e5df0539797c1d2a644307c9cd45976087bf4` |
| H10 N=10 profile | `a50d820823c5fa7a1aaac9cedaa1c3ef4af85b891f3962b1400ec0ded558df23` |
| 500-episode ledger | `8c83e6510a470a67d99cd66ca54bdde29113db08f95301a2e2a91b4e7453af4a` |
| 500-episode summary | `fde592fa6fe3b8353bbcd11ae580ecf3b718325f75d96f88ffd6ec99d7cb7acd` |

For this checkpoint and the README LIBERO-10 protocol, **10 balanced
Observations are sufficient to reproduce baseline task accuracy**. This is the
smallest task-balanced count tested, not proof that fewer than ten or a single
task's data would work. It is also not a claim that the absmax scales have
converged; business accuracy stabilized long before the per-site maxima did.
