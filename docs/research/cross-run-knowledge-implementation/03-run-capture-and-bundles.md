# M3 — run capture, quarantine, sanitation, and bundles

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1. Uses M2's publication boundary after its contract is stable.

## Objective

Turn one reconciled local evolve frontier into an immutable, sanitized
`RunBundle` without interpreting experiment value. Complete, stopped, and crashed
runs are harvestable only to the exact frontier their durable authorities can
jointly prove.

## Owned responsibilities

- Append-only execution-revision journal.
- Atomic capture generations and watermarks.
- Cross-artifact structural/provenance validation.
- Restricted local quarantine.
- Deterministic scope/task-family sanitation and taint report.
- Immutable sanitized `RunBundle` assembly/publication.
- Supersession of older captures from the same run.

## Proposed code surface

```text
src/kapso/cross_run/capture/
  __init__.py
  journal.py
  exporter.py
  validator.py
  sanitation.py
  bundle.py

src/kapso/execution/
  orchestrator.py

src/kapso/execution/memories/experiment_memory/
  store.py

tests/
  test_execution_revision_journal.py
  test_cross_run_capture_exporter.py
  test_cross_run_capture_validator.py
  test_cross_run_sanitation.py
  test_cross_run_bundle.py
```

## Execution-revision journal

The current experiment store is a latest executed projection. Cross-run evidence
also needs failed/interrupted revisions that a later recovery replaces.

- [ ] Add one strict append-only journal event per execution revision before the
      latest projection is replaced.
- [ ] Record run/campaign/node/idea/batch identity, revision, timestamps,
      execution/evaluation state, feedback, technical difficulties, evaluator
      fingerprint, measurements, and exact artifact refs.
- [ ] Require per-node gap-free revisions and idempotent identical replay.
- [ ] Reject the same node/revision with different content.
- [ ] Keep the journal local and current-run-only; it is not a cross-run store.
- [ ] Reconcile journal terminal revisions with `ExperimentHistoryStore`,
      `IdeaArchive`, node history, and checkpoint before capture.

M3 owns the necessary orchestrator/store write-order changes. M6 must not modify
this ordering.

## Capture generation

At a safe checkpoint frontier, `RunCaptureExporter` records:

```text
capture identity and generation
run/campaign/scope/launch identities
checkpoint frontier and strategy watermarks
IdeaArchive revision
ExperimentHistoryStore revision/count
execution-journal watermark
node/branch frontier
artifact completeness declarations
content refs and checksums
superseded capture generation
```

- [ ] Export only after local checkpoint/archive/store reconciliation succeeds.
- [ ] Stage one complete manifest plus referenced files on the same filesystem,
      flush, and atomically publish the generation marker.
- [ ] Never infer absent content: mark it `present`, `absent_before_frontier`, or
      `unavailable` under strict rules.
- [ ] Permit periodic capture after a durable checkpoint and final capture after
      a normal/stopped run; no asynchronous snapshot of mutating stores.
- [ ] Preserve the last committed capture if the next generation is interrupted.
- [ ] Keep branch/source exports path-allowlisted and tied to exact Git commits.

## Validation

`CaptureValidator` is deterministic and checks:

- schemas, IDs, hashes, and launch/scope identity;
- checkpoint, archive, experiment, journal, node, and branch watermarks;
- one selected idea per node and one node per executed idea;
- parent/node/branch/diff/evaluator provenance;
- objective direction and normalized utility;
- evaluation fingerprint completeness and measurement validity;
- attempt revision ordering and terminal projection agreement;
- completeness declarations; and
- source tree versus evaluated intervention identity.

Malformed/corrupt content raises. Validation never skips a bad record or repairs
foreign output.

## Quarantine and sanitation

Raw capture first enters an access-restricted, deletable local quarantine root
from config. `SanitationGate` emits only allowlisted content and a signed/reportable
sanitation result.

- [ ] Exclude `.env`, credentials, Git credential/config material, VCS history,
      caches, datasets, model weights, hidden evaluator material, raw task output,
      and unapproved logs before durable publication.
- [ ] Validate archive paths before extraction/copy; reject traversal, device
      files, unexpected symlinks, and submodules.
- [ ] Apply exact scope/task-family secret, identity, contamination, license,
      dependency, and artifact-class policies.
- [ ] Retain safe structured observations and content-addressed allowlisted source
      artifacts only.
- [ ] Record scanner/policy versions, findings, excluded paths/classes, and taint
      sources.
- [ ] An optional coding-agent semantic sweep may only escalate surviving content
      for review; deterministic rejection cannot be overridden.
- [ ] Delete raw quarantine according to configured retention only after bundle
      publication/verification or explicit rejection recording.

## Bundle publication

`RunBundlePublisher`:

- [ ] Builds the exact M1 `RunBundle` manifest from sanitized content.
- [ ] Content-addresses referenced blobs and avoids duplicate manifest payloads.
- [ ] Preserves completion state and the honest capture frontier.
- [ ] Publishes a later capture as a superseding bundle, never an overwrite.
- [ ] Uses M2 to submit/publish the knowledge-repository delta; it never performs
      GitHub calls itself.
- [ ] Does not label outcomes positive/negative, propose claims, or trigger expert
      evolution.

## Tests

- Capture complete, budget-stopped, recoverable-failure, and hard-crash fixtures.
- Reconcile multiple attempts of one idea into a gap-free journal.
- Inject interruption at each journal and capture durable write boundary.
- Reject mixed-generation store/archive/checkpoint/branch inputs.
- Reject unmeasured-baseline effects and invalid evaluator fingerprints.
- Verify all secret/evaluator/path/license fixtures are excluded or rejected by
  explicit policy.
- Prove model weights/data/raw logs never enter sanitized output by default.
- Prove repeated final capture is idempotent and a later frontier supersedes rather
  than double-counts the earlier one.
- Round-trip a bundle through M2's fake publication/materialization boundary.

## Definition of done

- One safe checkpoint frontier produces one independently verifiable bundle.
- A crash never exposes a mixed or partial capture generation.
- Earlier failed attempts survive recovery in the journal and later episode.
- Raw quarantine and sanitized durable output are physically and logically
  separate.
- The module makes no scientific claim and calls no generative model API.

## Non-goals

- Episode/prior-idea projection.
- Review, claim generation, or admission.
- Snapshot indexing or retrieval.
- Expert candidate creation.
