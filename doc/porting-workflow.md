# VLA Porting Workflow

Intake, trusted-source inspection, and Capability Contract classification are
available through one command surface: `scripts/apxinf_port.py`. It creates and
validates private Port artifacts without modifying the ApxInf source tree.
Source-model code runs only when a Reference Adapter entrypoint and dependency
lock are explicitly configured.

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
Every new request pins the exact `major.minor` Capability Contract version used
by Preflight. The shipped default is `1.0`.

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
`postprocess(output)`. It must also expose `describe()` and explicitly report
operator traces, preprocessing, tokenization, normalization, stochastic inputs,
schedules, custom operators, and dynamic branches; an empty list is the explicit
declaration that a list-valued feature is absent.

`describe()` also supplies `capability_facts`, with one explicit observation for
each required semantic dimension: shape profiles, attention, masks, position
encodings, normalization, activations, conditioning, action heads, schedules,
and control flow. Values are compared to the machine-readable contract rather
than inferred from model names.

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

## Run Intake and Preflight

```bash
python3 scripts/apxinf_port.py run --port-dir /private/path/my-port
python3 scripts/apxinf_port.py report --port-dir /private/path/my-port
```

`run` always attempts to write `report.json`, including for malformed,
unsupported, and Reference Adapter failures. The report keeps
`request_declarations` separate from machine-observed `observed_environment`
facts, records omitted facts under `warnings`, and records every supported tuple
as either `requested` or `not_requested`.

When source inspection is configured, `run` classifies every declared
capability as `supported`, `canonicalizable`, or `unsupported`. Unknown,
contradictory, undeclared, or unexplained semantics block Preflight and produce
a structured Gap Report. Contract revisions are declared as `initial`,
`additive`, or `breaking`: additive changes increment the minor version, while
changed or removed semantics require a new major version. Per-capability hashes
let later resume logic invalidate only evidence that depends on changed
capabilities.

Configured inspection generates only private Port artifacts:

- `private/reference_adapter.py`: the generated stable adapter contract
- `private/reference_environment/environment.json`: locked environment evidence
- `private/source_inventory.json`: reproducible architecture and semantics inventory
- `private/captures/inspection.json`: private inputs, outputs, and intermediates
- `private/capability_classification.json`: classification and dependency hashes
- `private/capability_gap_report.json`: unsupported semantics when Preflight blocks

The inventory and report bind these artifacts to the declared source revision,
source digest, and checkpoint digest. Port directories are rejected when they
are inside either ApxInf or the trusted source checkout, so adapters and captured
inputs cannot enter a proposed public commit. Tensor captures contain f32 data
with explicit source dtype and shape metadata. Report artifact records carry
content, workflow-tool, source, checkpoint, environment, and upstream-request
fingerprints. Dependency-aware stale retention and resume are added by the later
workflow-resume stage.

| Exit code | Category | Meaning |
| ---: | --- | --- |
| 0 | `success` | Intake passed |
| 2 | `missing_input` | The request file or an explicitly declared input disappeared |
| 3 | `invalid_input` | JSON or request schema validation failed |
| 4 | `unsupported_target` | A target/precision tuple is unsupported |
| 5 | `environment_failure` | The environment could not be observed or verified |
| 6 | `reference_load_failure` | The trusted source or checkpoint could not load |
| 7 | `reference_trace_failure` | Preprocessing, inference, capture, postprocessing, or inventory failed |
| 8 | `unsupported_semantics` | Source semantics are unknown, contradictory, unexplained, or outside the pinned contract |

The versioned contracts are
[`schemas/port-request-v1.schema.json`](../schemas/port-request-v1.schema.json)
and [`schemas/port-report-v1.schema.json`](../schemas/port-report-v1.schema.json).
Reference inspection adds
[`schemas/reference-inventory-v1.schema.json`](../schemas/reference-inventory-v1.schema.json),
[`schemas/reference-environment-v1.schema.json`](../schemas/reference-environment-v1.schema.json),
and [`schemas/reference-capture-v1.schema.json`](../schemas/reference-capture-v1.schema.json).
Capability classification uses
[`contracts/vla-capability-contract-1.0.json`](../contracts/vla-capability-contract-1.0.json),
validated by
[`schemas/vla-capability-contract-v1.schema.json`](../schemas/vla-capability-contract-v1.schema.json).
Its result and terminal gap formats are
[`schemas/capability-classification-v1.schema.json`](../schemas/capability-classification-v1.schema.json)
and
[`schemas/capability-gap-report-v1.schema.json`](../schemas/capability-gap-report-v1.schema.json).
