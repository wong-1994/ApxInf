# VLA Porting Workflow

Intake and trusted-source inspection are available through one command surface:
`scripts/apxinf_port.py`. It creates and validates private Port artifacts without
modifying the ApxInf source tree. Source-model code runs only when a Reference
Adapter entrypoint and dependency lock are explicitly configured.

## Initialize a request

Keep the Port directory outside the repository because it contains private local
paths and may later contain reference evidence.

```bash
python3 scripts/apxinf_port.py init \
  --port-dir /private/path/my-port
```

`--source`, `--source-revision`, and `--checkpoint` may be supplied during
initialization or left blank in the draft. When paths are supplied,
initialization verifies them, hashes the source tree and checkpoint, and Intake
recomputes those hashes so later results cannot silently use changed inputs.

All model and qualification facts are optional at Intake. Missing facts produce
warnings and reduce the guarantees available to later stages; they do not block
the Port. In particular, missing latency goals or tuning budgets means
performance is not guaranteed. Fill in representative input shapes, requested
target/precision tuples, p50 and p95 latency goals, correctness thresholds, and
per-target tuning budgets whenever those guarantees are required. Valid tuples
are:

- Thor: `bf16`, `fp8`
- Orin: `bf16`, `int8_w8a8`

Orin `fp8` is rejected during Intake.

## Configure trusted-source inspection

The trusted source opts into the stable Reference Adapter contract by exposing
five module-level callables: `load(checkpoint_path)`, `preprocess(profile)`,
`infer(model, inputs)`, `capture_intermediates(model, inputs)`, and
`postprocess(output)`. It may also expose `describe()` to report operator traces,
preprocessing, tokenization, normalization, stochastic inputs, schedules, custom
operators, and dynamic branches.

Both paths below are relative to the trusted source root:

```bash
python3 scripts/apxinf_port.py init \
  --source /trusted/model-source \
  --source-revision 0123456789abcdef \
  --checkpoint /trusted/checkpoints/model.ckpt \
  --reference-entrypoint reference_impl.py \
  --dependency-lock requirements.lock \
  --port-dir /private/path/my-port
```

The dependency lock is installed with pip's `--require-hashes` and `--no-index`
inside a Port-local virtual environment. Empty locks are valid for sources that
use only the Python standard library. Runtime source execution receives offline
library settings and a Python socket guard; this protects against accidental
access by trusted code and is not a sandbox for malicious code.

## Run Intake and inspect the report

```bash
python3 scripts/apxinf_port.py run --port-dir /private/path/my-port
python3 scripts/apxinf_port.py report --port-dir /private/path/my-port
```

`run` always attempts to write `report.json`, including for malformed,
unsupported, and Reference Adapter failures. The report keeps
`request_declarations` separate from machine-observed `observed_environment`
facts, records omitted facts under `warnings`, and records every supported tuple
as either `requested` or `not_requested`.

Configured inspection generates only private Port artifacts:

- `private/reference_adapter.py`: the generated stable adapter contract
- `private/reference_environment/environment.json`: locked environment evidence
- `private/source_inventory.json`: reproducible architecture and semantics inventory
- `private/captures/inspection.json`: private inputs, outputs, and intermediates

The inventory and report bind these artifacts to the declared source revision,
source digest, and checkpoint digest. Port directories are rejected when they
are inside either ApxInf or the trusted source checkout, so adapters and captured
inputs cannot enter a proposed public commit.

| Exit code | Category | Meaning |
| ---: | --- | --- |
| 0 | `success` | Intake passed |
| 2 | `missing_input` | The request file or an explicitly declared input disappeared |
| 3 | `invalid_input` | JSON or request schema validation failed |
| 4 | `unsupported_target` | A target/precision tuple is unsupported |
| 5 | `environment_failure` | The environment could not be observed or verified |
| 6 | `reference_load_failure` | The trusted source or checkpoint could not load |
| 7 | `reference_trace_failure` | Preprocessing, inference, capture, postprocessing, or inventory failed |

The versioned contracts are
[`schemas/port-request-v1.schema.json`](../schemas/port-request-v1.schema.json)
and [`schemas/port-report-v1.schema.json`](../schemas/port-report-v1.schema.json).
Reference inspection adds
[`schemas/reference-inventory-v1.schema.json`](../schemas/reference-inventory-v1.schema.json),
[`schemas/reference-environment-v1.schema.json`](../schemas/reference-environment-v1.schema.json),
and [`schemas/reference-capture-v1.schema.json`](../schemas/reference-capture-v1.schema.json).
