# Model Porting Workflow

The family-neutral Porting Core exposes Intake, trusted-source inspection,
Capability Contract classification, and Canonical equivalence through one
command surface:
`scripts/apxinf_port.py`. It creates and validates private Port artifacts
without modifying the ApxInf source tree. Source-model code runs only when a
Reference Adapter entrypoint and dependency lock are explicitly configured.

## Initialize a request

Keep the Port directory outside the repository because it contains private local
paths and may later contain reference evidence.

```bash
python3 scripts/apxinf_port.py init \
  --family vla \
  --port-dir /private/path/my-port
```

Every request explicitly selects `llm`, `vlm`, or `vla`. The selected Family
Pack and its exact Capability Contract version are pinned before trusted source
code can execute. The VLA Family Pack is the first registered implementation;
LLM and VLM requests are recognized but rejected until their packs are added.

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
than inferred from model names. Preflight reconciles these declarations with
the captured input schemas, `normalization.model`, each schedule `kind`, each
dynamic-branch `kind`, and any `operator_traces[].semantic_capabilities`
annotations, so contradictory raw inventory cannot pass by assertion alone.

Both paths below are relative to the trusted source root:

```bash
python3 scripts/apxinf_port.py init \
  --family vla \
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

## Prove canonical equivalence

Every source that passes capability classification emits the same versioned
Canonical VLA trace contract. A source whose semantics are already canonical
uses direct mode and does not create a Canonical Adapter. A source with one or
more `canonicalizable` capabilities must additionally expose
`canonicalize(model)`, `canonical_preprocess(profile)`,
`canonicalize_preprocessed_inputs(inputs)`, `canonical_infer(model, inputs)`,
`canonical_capture_intermediates(model, inputs)`,
`canonical_postprocess(output)`, and `canonicalization_manifest()` from its
trusted entrypoint. The generic Canonical Adapter wrapper is copied into the
private Port directory only for that case.

The manifest consumes every named source and canonical parameter exactly once,
with shape, dtype, alias, and tied-weight checks, and declares all transpose,
split, concatenation, packing, mask, conditioning, cache, and schedule
transformations that apply. Each transformation is labeled
`algebraic` or `numerical_equivalence`; algebraic transformations must list
their assumptions. The manifest also maps preprocessing representations and
selected intermediate checkpoints,
accounts for source branches, and covers each declared mask, conditioning,
cache, or schedule state transformation without inventing absent state.

The gate requires at least two distinct representative profiles and explicit
absolute and relative thresholds. It resets Python, NumPy, PyTorch, and an
optional source `set_seed(seed)` hook for seeds 0 and 1. It compares canonical
preprocessing against the declared source-to-canonical input transform, then
compares selected intermediates, normalized actions, and deployable
postprocessed actions for every profile/seed case. An incomplete manifest or
failed comparison blocks Preflight with a private Canonical VLA Gap Report.

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
The family-neutral Porting Core owns these lifecycle states, stable exit
categories, requested-tuple states, and named Gate results. Family Packs return
semantic evidence to those Gates without teaching the Core about actions,
logits, media, caches, or schedules.

When source inspection is configured, `run` classifies every declared
capability as `supported`, `canonicalizable`, or `unsupported`. Unknown,
contradictory, undeclared, or unexplained semantics block Preflight and produce
a structured Gap Report. Contract revisions are declared as `initial`,
`additive`, or `breaking`: additive changes increment the minor version, while
changed or removed semantics require a new major version. The prior contract is
loaded and compared rather than trusting revision labels. Per-capability hashes
and `invalidated_capabilities` identify only dependencies affected by an update.
Passing capability classification advances through canonical equivalence
and family-neutral kernel coverage. Every canonical computation is classified
as existing fused, existing primitive, layout-only, correct fallback, missing
required capability, or unsupported. Unclassified computations fail closed; a
correct fallback becomes a non-blocking Optimization Opportunity.

A missing required capability blocks Preflight before model implementation and
emits `private/kernel_gap_handoff.json` for the separate kernel workflow. The
handoff preserves family semantics and references plus dtype, layout, shapes,
tolerances, golden tensors, requested targets, frequency, performance impact,
and expected interface. A capability returned by that workflow must record
successful revalidation against the original family references before a rerun
can pass Preflight.

Configured inspection generates only private Port artifacts:

- `private/reference_adapter.py`: the generated stable adapter contract
- `private/reference_environment/environment.json`: locked environment evidence
- `private/source_inventory.json`: reproducible architecture and semantics inventory
- `private/captures/inspection.json`: private inputs, outputs, and intermediates
- `private/capability_classification.json`: classification and dependency hashes
- `private/capability_gap_report.json`: unsupported semantics when Preflight blocks
- `private/canonical_adapter.py`: generated only for a canonicalized source
- `private/canonical_trace.json`: downstream trace for direct or canonicalized sources
- `private/canonical_equivalence.json`: parameter, rewrite, and comparison evidence
- `private/canonicalization_gap_report.json`: incomplete or failed equivalence evidence
- `private/kernel_coverage.json`: classifications and Optimization Opportunities
- `private/kernel_gap_handoff.json`: complete blocking Kernel Gap requirements

The inventory and report bind these artifacts to the declared source revision,
source digest, and checkpoint digest. Port directories are rejected when they
are inside either ApxInf or the trusted source checkout, so adapters and captured
inputs cannot enter a proposed public commit. Tensor captures contain f32 data
with explicit source dtype and shape metadata. Report artifact records carry
content, workflow-tool, source, checkpoint, ApxInf source, kernel build,
Capability Contract, documentation, applicable target-environment, and upstream
fingerprints through a versioned common envelope that also pins the selected
Family Pack, Capability Contract, stage, and family payload schema. The Family
Pack validates each payload before its envelope is emitted.

## Resume safely

```bash
python3 scripts/apxinf_port.py resume --port-dir /private/path/my-port
```

`resume` recomputes dependency and payload fingerprints; file existence alone
never makes evidence current. Changed artifacts and their named descendants are
marked `stale` but retained for diagnosis. Gates backed by stale evidence also
become stale. Unaffected family and common artifacts remain current. A stage
left `running` by interruption is reset deterministically to `not_started`, and
the report records the interrupted stages, stale artifacts, and a structured
resumption explanation.

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
| 9 | `correctness_failure` | Canonicalization evidence is incomplete or a differential comparison failed |
| 10 | `kernel_gap` | A required capability is handed to the separate kernel workflow |

The versioned contracts are
[`schemas/port-request-v1.schema.json`](../schemas/port-request-v1.schema.json)
and [`schemas/port-report-v1.schema.json`](../schemas/port-report-v1.schema.json).
Artifact records use
[`schemas/workflow-artifact-envelope-v1.schema.json`](../schemas/workflow-artifact-envelope-v1.schema.json).
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
Canonical evidence uses
[`schemas/canonical-trace-v1.schema.json`](../schemas/canonical-trace-v1.schema.json),
[`schemas/canonical-equivalence-v1.schema.json`](../schemas/canonical-equivalence-v1.schema.json),
and
[`schemas/canonicalization-gap-report-v1.schema.json`](../schemas/canonicalization-gap-report-v1.schema.json).
Kernel coverage uses
[`schemas/kernel-computation-v1.schema.json`](../schemas/kernel-computation-v1.schema.json),
[`schemas/kernel-coverage-v1.schema.json`](../schemas/kernel-coverage-v1.schema.json),
and
[`schemas/kernel-gap-handoff-v1.schema.json`](../schemas/kernel-gap-handoff-v1.schema.json).

## Family-neutral GEMM tuning

Family Packs export execution plans through `scripts/tuning_workloads.py` as
`apxinf.tuning.gemm-workloads.v1`. Each entry is a physical GEMM contract plus
family, logical phase (`vision`, `prefill`, `decode`, or `action`), source
operation, profile, executions per inference, and a complete target
fingerprint. The generic tuner never imports a model or Family Pack config.

The tuner orders work by estimated milliseconds saved per execution multiplied
by executions per inference, enforces a budget for each device/profile pair,
and reports coverage, best current results, and remaining hotspots. Persisted
tactics are accepted only when device name, SM, multiprocessor count, kernel
build, CUDA version, and every recorded library version match. After tactics
are installed, the caller must rerun the selected Family Pack's complete
inference correctness check. Version 1 intentionally rejects non-GEMM tunable
objects.
