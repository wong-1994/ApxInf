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
  --model-dir "$APXINF_MODEL_DIR" \
  --manifest /path/to/observations.jsonl
```

By default this writes `$APXINF_MODEL_DIR/calibration.json`. Use `--output` for
another location. Existing files are not overwritten unless `--force` is
passed.

Choose observations that represent deployment cameras, prompts, robot state,
lighting, scenes, and object poses. Every observation must contain exactly the
camera views expected by the checkpoint.

## LIBERO and other LeRobot datasets

For the PI0.5 LIBERO checkpoint, a profile can be generated directly from the
LeRobot LIBERO dataset:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir "$APXINF_MODEL_DIR" \
  --dataset lerobot/libero
```

This adapter requires LeRobot. It is not limited to LIBERO: `--dataset` accepts
any compatible LeRobot repository ID. `--dataset-root PATH` selects an existing
local copy. Sampling is deterministic and task-balanced; by default it selects
one frame per `task_index`, while `--samples N` sets the total count. A dataset
without `task_index` requires an explicit `--samples N`.

LeRobot is only a data-source adapter. `Pi05CalibrationJob` itself consumes an
iterable of model-native Observation dictionaries and has no LeRobot dependency.

## Replay existing NPZ observations

Existing automation can pass a directory or repeat individual inputs:

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir "$APXINF_MODEL_DIR" \
  --input-dir /path/to/observation-npz
```

```bash
python3 scripts/calibrate_pi05.py \
  --model-dir "$APXINF_MODEL_DIR" \
  --input sample-000.npz \
  --input sample-001.npz
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

Run `python3 scripts/calibrate_pi05.py --help` for checkpoint-specific input
overrides and reproducibility options.
