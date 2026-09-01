# PI0.5 FP8 calibration

Static FP8 needs representative activation scales in
`<model-dir>/calibration.json`. The calibration input uses the same public
Observation fields as inference; do not provide preprocessed tensors such as
`rgb`, `token_ids`, or `noise`.

## Observation manifest

The portable input is a JSONL file with one Observation per line. Image values
are paths relative to the manifest, or absolute paths. State is optional unless
the checkpoint's input configuration requires it.

```json
{"observation/image":"frames/000-base.png","observation/wrist_image":"frames/000-wrist.png","prompt":"pick up the block","observation/state":[0.1,0.2,0.3]}
{"observation/image":"frames/001-base.png","observation/wrist_image":"frames/001-wrist.png","prompt":"open the drawer","observation/state":[0.2,0.1,0.4]}
```

Generate the profile:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --manifest <path-to-observations.jsonl>
```

By default this writes `<path-to-model>/calibration.json`. Use `--output` for
another location. Existing files are not overwritten unless `--force` is
passed.

Choose observations that represent deployment cameras, prompts, robot state,
lighting, scenes, and object poses. Every observation must contain exactly the
camera views expected by the checkpoint.

## Native LIBERO task observations

For the PI0.5 LIBERO checkpoint, generate a profile from observations rendered
by the actual LIBERO10 task suite:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --libero-suite libero_10
```

This path uses LIBERO's BDDL tasks, language instructions, initial states, and
off-screen simulator cameras. It applies the same camera orientation, resize,
and robot-state conversion as `scripts/eval_libero.py`. Sampling is
deterministic and task-balanced. By default it captures one settled initial
state from every task; `--samples N` selects more initial states while retaining
balanced task coverage.

This command needs the same LIBERO and MuJoCo dependencies as the repository's
LIBERO evaluation command. It does not require LeRobot or a downloaded replay
dataset. For another simulator or deployment source, export its public
Observations through the manifest interface instead.

## Replay existing NPZ observations

Existing automation can pass a directory or repeat individual inputs:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --input-dir <path-to-observation-npz>
```

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir <path-to-model> \
  --input <path-to-sample-000.npz> \
  --input <path-to-sample-001.npz>
```

Each NPZ contains the configured image keys, scalar prompt string, and optional
state. NPZ is useful for exact replay; it is not required for normal calibration.

## Loading the profile

`AutoPolicy` automatically uses `<model-dir>/calibration.json`:

```python
policy = AutoPolicy.from_pretrained("<path-to-model>", precision="fp8")
```

Pass `calibration=` only when the profile is stored elsewhere:

```python
policy = AutoPolicy.from_pretrained(
    "<path-to-model>",
    precision="fp8",
    calibration="/path/to/calibration.json",
)
```

At FP8 startup, ApxInf reads the checkpoint shards to compare their SHA-256
identity with the profile. This can add startup time on slower storage. An
identity mismatch emits a warning and continues; malformed profiles, missing or
invalid scales, and incompatible execution plans still fail because the runtime
cannot use them safely.

`tactics.json` is optional. If it is absent, the runtime warns and continues
with compatible kernel fallbacks; calibration is unaffected, though performance
may be lower than with tuned tactics.

Profiles using an older calibration schema are not migrated automatically.
Regenerate `calibration.json` with the ApxInf version being deployed.

Run `python3 scripts/calibrate_pi05.py --help` for checkpoint-specific input
overrides and reproducibility options.
