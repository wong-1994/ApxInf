# VLA Porting Workflow

The first workflow stage, Intake, is available through one command surface:
`scripts/apxinf_port.py`. It creates and validates private Port artifacts without
modifying the ApxInf source tree or executing source-model code.

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

## Run Intake and inspect the report

```bash
python3 scripts/apxinf_port.py run --port-dir /private/path/my-port
python3 scripts/apxinf_port.py report --port-dir /private/path/my-port
```

`run` always attempts to write `report.json`, including for malformed and
unsupported requests. The report keeps `request_declarations` separate from
machine-observed `observed_environment` facts, records omitted facts under
`warnings`, and records every supported tuple as either `requested` or
`not_requested`.

| Exit code | Category | Meaning |
| ---: | --- | --- |
| 0 | `success` | Intake passed |
| 2 | `missing_input` | The request file or an explicitly declared input disappeared |
| 3 | `invalid_input` | JSON or request schema validation failed |
| 4 | `unsupported_target` | A target/precision tuple is unsupported |
| 5 | `environment_failure` | The environment could not be observed or verified |

The versioned contracts are
[`schemas/port-request-v1.schema.json`](../schemas/port-request-v1.schema.json)
and [`schemas/port-report-v1.schema.json`](../schemas/port-report-v1.schema.json).
