# VLA Porting Workflow

The first workflow stage, Intake, is available through one command surface:
`scripts/apxinf_port.py`. It creates and validates private Port artifacts without
modifying the ApxInf source tree or executing source-model code.

## Initialize a request

Keep the Port directory outside the repository because it contains private local
paths and may later contain reference evidence.

```bash
python3 scripts/apxinf_port.py init \
  --source /path/to/trusted/source \
  --source-revision <commit-or-release-id> \
  --checkpoint /path/to/checkpoint \
  --port-dir /private/path/my-port
```

Initialization verifies that the source directory and checkpoint exist, hashes
the source tree and checkpoint, and writes `request.json`. Intake recomputes
those hashes so later results cannot silently use changed inputs. Fill in the
draft's representative input shapes, requested target/precision tuples, p50 and
p95 latency goals, correctness thresholds, and per-target tuning budgets. Valid
tuples are:

- Thor: `bf16`, `fp8`
- Orin: `bf16`, `int8_w8a8`

Orin `fp8` is rejected during Intake.

## Run Intake and inspect the report

```bash
python3 scripts/apxinf_port.py run --port-dir /private/path/my-port
python3 scripts/apxinf_port.py report --port-dir /private/path/my-port
```

`run` always attempts to write `report.json`, including for missing, malformed,
and unsupported requests. The report keeps `request_declarations` separate from
machine-observed `observed_environment` facts and records every supported tuple
as either `requested` or `not_requested`.

| Exit code | Category | Meaning |
| ---: | --- | --- |
| 0 | `success` | Intake passed |
| 2 | `missing_input` | A required fact or file is absent |
| 3 | `invalid_input` | JSON or request schema validation failed |
| 4 | `unsupported_target` | A target/precision tuple is unsupported |
| 5 | `environment_failure` | The environment could not be observed or verified |

The versioned contracts are
[`schemas/port-request-v1.schema.json`](../schemas/port-request-v1.schema.json)
and [`schemas/port-report-v1.schema.json`](../schemas/port-report-v1.schema.json).
