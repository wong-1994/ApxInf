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
code can execute. The VLA and LLM Family Packs are registered; VLM requests are
recognized but rejected until that pack is added.

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
each required semantic dimension. VLA covers shape profiles, attention, masks,
positions, normalization, activations, conditioning, action heads, schedules,
and control flow. LLM covers text shape profiles, tokenizer/chat templates,
embeddings, attention and masks, positions, normalization, activations, KV
cache, generation state, sampling, and control flow. Values are compared to the
machine-readable contract rather than inferred from model names. Preflight
reconciles these declarations with
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

The Agent prepares the reference environment; `apxinf_port.py` does not choose
an installer, create a venv, start a container, or decide whether network access
is appropriate. Prefer, in order: a previously verified target container or
venv, an offline install from available caches, then a lock-respecting install
with explicitly authorized network access. Record the selected environment with
`scripts/record_reference_environment.py`, executed by that environment's
Python. The recorder hashes the lock and inventories the interpreter and
installed distributions; it does not install or mutate anything.

Run `apxinf_port.py run` once after completing the request to materialize
`private/reference_profiles.json`; a missing-evidence result is an instruction
to the Agent, not a failed Port. The Agent then invokes the generated private
`reference_adapter.py` with its selected Python and writes these fixed evidence
paths:

- `private/reference_environment/environment.json`
- `private/source_inventory.json`
- `private/captures/inspection.json`
- `private/inspection_result.json`

The adapter command takes `--source-root`, `--entrypoint`, `--checkpoint`,
`--profiles`, the three output paths, and the source/checkpoint revision and
digest values pinned in `request.json`. Environment-specific asset paths and
container mounts are Agent decisions and remain outside the request schema.
After evidence exists, rerun `apxinf_port.py run`; it only validates provenance,
schemas, capability facts, numerical evidence, and Gates.

If offline preparation lacks a wheel or model asset, diagnose the missing item
and continue with another existing environment or request authorization for a
lock-respecting download. Do not encode the encountered package manager or
machine layout as a new Porting Core branch.

## Prove canonical equivalence

Every source that passes capability classification emits the same versioned
named-tensor canonical trace contract with its selected family. A source whose
semantics are already canonical
uses direct mode and does not create a Canonical Adapter. A source with one or
more `canonicalizable` capabilities must additionally expose
`canonicalize(model)`, `canonical_preprocess(profile)`,
`canonicalize_preprocessed_inputs(inputs)`, `canonical_infer(model, inputs)`,
`canonical_capture_intermediates(model, inputs)`,
`canonical_postprocess(output)`, and `canonicalization_manifest()` from its
trusted entrypoint. The generic Canonical Adapter wrapper is copied into the
private Port directory only for that case.

The Porting Core does not execute it. A missing-canonical-evidence result tells
the Agent to run `private/canonical_adapter.py` with the already selected
reference environment, then rerun the Gate. The fixed outputs are
`private/canonical_trace.json`, `private/canonical_equivalence.json`, and
`private/canonical_result.json`.

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
compares family-required checkpoints and final outputs for every profile/seed
case. For LLM this includes tokenizer output, representative layers, prefill and
decode logits, KV-cache values and positions, reset state, generated tokens, and
EOS handling. An incomplete manifest or failed comparison blocks Preflight with
a private canonicalization Gap Report.

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

Layout-only rewrites and correct fallbacks require operator-replay evidence in
the canonical computation. The replay must cite the computation's original
family references and report absolute and relative errors within its declared
tolerances. A catalog entry alone cannot establish either classification;
missing, mismatched, self-failed, or out-of-tolerance replay becomes a blocking
Kernel Gap. Existing primitives still require complete semantic matching in the
catalog; broad operator-name similarity is not evidence that masks, layouts, or
shape constraints agree.

A missing required capability blocks Preflight before model implementation and
emits `private/kernel_gap_handoff.json` for the separate kernel workflow. The
handoff preserves family semantics and references plus dtype, layout, shapes,
tolerances, golden tensors, requested targets, frequency, performance impact,
and expected interface. A capability returned by that workflow must record
successful revalidation against the original family references before a rerun
can pass Preflight.

The Agent consumes that handoff immediately, follows `adding-new-kernels.md`,
and reruns the same Port after the returned capability is reference-validated.
A Kernel Gap is a workflow transition, not a reason to ask the user what to do,
unless implementation needs new authority or unavailable hardware.

### Recover when implementation invalidates Preflight

Preflight is provisional until maintained implementation exercises the claimed
ApxInf interfaces. If source inspection, operator replay, Backend inspection, or
model implementation shows that a passed computation was omitted,
misclassified, bound to an unrelated tensor, or cannot be realized by the
claimed ApxInf capability, invalidate the affected coverage result immediately.
This discovery is an evidence transition even when the previous report contains
no `kernel_gap_handoff.json`.

Continue the same task in this order:

1. Mark the affected Preflight evidence stale and correct the Family Adapter's
   operator trace with operation-local inputs, parameters, outputs, semantics,
   dtype, layout, and concrete shapes.
2. Build the operator gap table required by `adding-new-kernels.md`. Classify
   each computation as directly reusable, expressible by existing primitives,
   missing a Backend operation, or requiring a new hardware implementation.
3. Rerun kernel coverage to materialize the corrected
   `private/kernel_gap_handoff.json`. If the current tooling cannot materialize
   the handoff, record the same required fields in the private Port and repair
   that workflow seam; absence of the old artifact does not end the Port.
4. Execute `adding-new-kernels.md` immediately for every genuine gap, validate
   the returned capability against the original operation-local reference
   tensors, and rerun the same Port.
5. Resume maintained model implementation and repeat this recovery loop whenever
   later layer or end-to-end comparisons invalidate another capability claim.

Finding more work is progress, not completion. A model-port request completes
only when the maintained observation-to-output path passes target-hardware
layerwise and end-to-end differential validation plus its requested serving
smoke tests, or when progress requires new authority or unavailable hardware.
In the latter case, report the concrete external blocker and the last passing
evidence. A discovered Kernel Gap, missing Backend API, failed implementation,
or absent handoff remains executable work inside the current task.

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

## Carry evidence between trusted machines

Create a local directory bundle, copy it through a trusted private channel, and
merge it into another directory for the same Port:

```bash
python3 scripts/apxinf_port.py bundle \
  --port-dir /private/path/my-port \
  --output /private/path/my-port-bundle
python3 scripts/apxinf_port.py merge-bundle \
  --port-dir /private/path/my-port \
  --bundle /private/path/my-port-bundle
```

`manifest.json` binds every included file to a SHA-256 digest and preserves the
Port, Family Pack, contract, source, checkpoint, dependency, and target-machine
provenance. Requests and artifact envelopes are stripped of absolute paths.
Credentials, checkpoint files, captured real inputs, and JSON payloads carrying
absolute paths or secret fields are omitted. Private Reference and Canonical
Adapters may remain in this non-publishable bundle for trusted verification.

Merge fails closed on tampering, conflicting artifact names, missing upstream
evidence, stale artifacts, family/payload mismatches, unrequested tuples, or
target evidence whose payload and target-environment fingerprint disagree.
Valid target evidence is copied into the Port and listed under
`portable_evidence`, allowing qualification to consume independently collected
requested tuples without treating tactics or performance as portable across
environments.

No command cleans up automatically. `cleanup` reports retention by default;
only `cleanup --confirm` removes the complete private Port directory. This is
irreversible unless the user made a separate backup.

## Prepare publication safely

Every Port report contains a `refactor_assessment`. It is explicitly `none`
when no shared architectural debt was found. Concrete debt is recorded as
`deferred` with a title, evidence, and proposed follow-up; it is never
implemented as part of the Port.

After the maintained implementation has been committed on a clean dedicated
branch named `port/<port-id>` (which may back a dedicated worktree),
`prepare-publication` validates the complete three-dot diff
from its pinned base commit. It rejects source/reference adapters, checkpoints,
real inputs, credentials, sensitive content, files without redistribution
approval, and files larger than 1 MiB. It neither stashes nor resets work.
The publication input must declare every changed path exactly once under
`public_files`, classify it as maintained source, a synthetic fixture, or
support metadata, and record explicit redistribution approval. Undeclared and
original-source material fails closed.

```bash
python3 scripts/apxinf_port.py prepare-publication \
  --port-dir /private/path/my-port \
  --repository /path/to/ApxInf \
  --base-commit <pinned-base-sha> \
  --publication /private/path/publication.json \
  --output /private/path/prepared-publication
```

The local output contains `support-metadata.json` with the family and only
requested tuples backed by Release-qualified evidence, plus
`pull-request.md`. Deferred debt also produces a submission-ready
`refactor-issue.md`, and the PR text always includes a Deferred Refactors
section. A remote action declaration (`push`, `create_pr`, `create_issue`, or
`link_issue`) is rejected unless `publication_authorized` is explicitly true;
preparation itself performs no remote action.

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

## VLA Thor FP8 qualification

`scripts/vla_fp8_qualification.py` qualifies only the tuples explicitly
requested by a VLA Port. A Thor FP8 request must identify its weights, scales,
and calibration inputs by SHA-256 and provide family-attributed kernel coverage
and tuning evidence. Tactics must match the validated Thor device, SM,
multiprocessor, kernel-build, CUDA, and library fingerprint.

The Gate machine-evaluates every declared VLA stage plus normalized and
deployable actions. User thresholds may be stricter than the VLA FP8 maxima,
but cannot weaken them. An FP8-only Port has no BF16 implementation or relative
Gate. When Thor BF16 is also requested, the declared minimum improvement is
evaluated against BF16 evidence from the identical device fingerprint. Public
support output contains only tuples that pass all applicable Gates; Orin FP8 is
rejected before evidence evaluation and can never be advertised.

## VLA Orin INT8 W8A8 qualification

`scripts/vla_int8_qualification.py` qualifies only the Orin tuples explicitly
requested by a VLA Port. An INT8 W8A8 request must identify its weights, scales,
and calibration inputs by SHA-256 and provide family-attributed kernel coverage
and tuning evidence. Tactics must match the validated Orin device, SM87,
multiprocessor, kernel-build, CUDA, and library fingerprint.

The Gate machine-evaluates every declared VLA stage plus normalized and
deployable actions against INT8-specific maxima. User thresholds may be stricter
than those maxima, but cannot weaken them. An INT8-only Port has no BF16
implementation or relative Gate. When Orin BF16 is also requested, the declared
minimum improvement is evaluated against BF16 evidence from the identical Orin
fingerprint. Public support output contains only requested tuples that pass all
applicable Gates; FP8 is rejected for Orin and can never be advertised.

## Common qualification state

`scripts/qualification.py` computes qualification state only across the tuples
declared by a Port request. Family Packs provide metric and deployment checks;
the Core stores their named Gates without knowing about control steps, tokens,
images, or other family concepts. The VLA adapter in
`scripts/vla_qualification.py` requires absolute control-step p50 and p95 limits
for every requested tuple.

Fresh tuple evidence records benchmark warmup and sample counts, observation
and action profiles, workspace and peak memory, and family-defined metrics.
Thor and Orin evidence also records power mode, clocks, temperature, device,
driver, CUDA, libraries, and kernel build. Nonconforming evidence remains in
diagnostics but cannot qualify a release. Missing representative real inputs
caps otherwise passing evidence at `provisional`.

Deployment-complete requires inference, VLA policy processing, and action
serving for every requested tuple. A Deployment-complete Port is
Performance-pending until all requested tuple Gates pass and is
Release-qualified only with fresh, conforming, representative evidence. A
same-device BF16 improvement Gate exists only when both BF16 and a lower
precision are requested for that target.

Correctness and performance waivers must be scoped to a requested tuple, cite
evidence, carry an ISO expiration date, and include explicit maintainer
approval. A waiver records an accepted deviation but cannot turn waived
evidence into Release-qualified evidence.

## VLA Family Pack acceptance

The VLA Family Pack binds seven explicit contracts to the shared Core:
capability classification, private reference capture, canonicalization,
named-tensor verification, the maintained `VlaRuntime` integration, action
serving, and the control-step benchmark profile. The deterministic public
acceptance fixture is software-validated with:

```bash
python3 scripts/vla_family_pack_acceptance.py \
  tests/fixtures/vla-family-pack-acceptance-v1.json
```

Acceptance replays the existing PI0.5 prompt, state-discretization,
reverse-time Euler-flow, policy-postprocessing, and serving behavior without
changing their mathematics. It also records a minimal synthetic external VLA
passing Intake, Preflight, maintained implementation, policy integration,
serving, tuning, qualification, bundling, and local PR preparation.

The software run cannot claim full acceptance. A controlled Thor runner must
repeat the immutable matrix and add the CUDA BF16 performance Gate:

```bash
python3 scripts/vla_family_pack_acceptance.py \
  tests/fixtures/vla-family-pack-acceptance-v1.json \
  --controlled-hardware --runtime-python python3 \
  --output vla-family-pack-acceptance-report.json
```

The report binds every stage to the canonical `synthetic-minimal-vla-v1`
subject, the exact Git commit, tool entry points, platform, and SHA-256 of every
public acceptance artifact. The command matrix is fixed in maintained code and
cannot be replaced by caller-supplied no-op checks.

Each successful command produces a lifecycle artifact containing the same
`port_id`, the command and output digests, and the preceding artifact digest.
Consequently Intake is the root of one tamper-evident chain through Preflight,
implementation, policy integration, serving, tuning, qualification, bundling,
and PR preparation; a collection of independently self-reported stage labels
cannot satisfy the chain. Existing PI0.5 prompt tokenization, state
discretization, and reverse-time Euler-flow replay is additionally stored with
the VLA Family Pack as a schema-validated shared-Core Workflow Artifact.

The fixture matrix fails closed for canonicalization and unsupported
semantics, distinguishes a blocking Kernel Gap from a non-blocking
Optimization Opportunity, prevents stale evidence from satisfying a Gate,
qualifies only requested target/precision tuples, and applies publication
safety. Public acceptance evidence contains only maintained source and
reviewed synthetic fixtures—never external source, private adapters,
checkpoints, real inputs, planning documents, or transient Port state.
Acceptance deliberately names no future production model and creates no
dependency from independently developed LLM or VLM Family Packs back to VLA.
