# M3 — run capture, quarantine, sanitation, and bundles

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1.

Status: **implemented; M9 runtime composition and M10 production activation remain**.

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
- Immutable local sanitized `RunBundle` assembly and content-addressed storage.
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
  pipeline.py
  evaluation_evidence.py
  git_evidence.py
  provenance.py
  safety.py

src/kapso/cross_run/
  git_command.py

src/kapso/execution/
  orchestrator.py

src/kapso/execution/memories/experiment_memory/
  store.py

tests/
  cross_run_capture_fixtures.py
  test_execution_revision_journal.py
  test_cross_run_capture_exporter.py
  test_cross_run_capture_validator.py
  test_cross_run_sanitation.py
  test_cross_run_bundle.py
  test_cross_run_capture_pipeline.py
  test_cross_run_git_command.py
```

## Execution-revision journal

The current experiment store is a latest executed projection. Cross-run evidence
also needs failed/interrupted revisions that a later recovery replaces.

- [x] Add one strict append-only journal event per execution revision before the
      latest projection is replaced.
- [x] Record run/campaign/node/idea/batch identity, revision, timestamps,
      execution/evaluation state, feedback, technical difficulties, evaluator
      fingerprint, measurements, and exact artifact refs.
- [x] Require per-node gap-free revisions and idempotent identical replay.
- [x] Reject the same node/revision with different content.
- [x] Keep the journal local and current-run-only; it is not a cross-run store.
- [x] Reconcile journal terminal revisions with `ExperimentHistoryStore`,
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

- [x] Export only after local checkpoint/archive/store reconciliation succeeds.
- [x] Stage one complete manifest plus referenced files on the same filesystem,
      flush, and atomically publish the generation marker.
- [x] Never infer absent content: mark it `present`, `absent_before_frontier`, or
      `unavailable` under strict rules.
- [x] Permit periodic capture after a durable checkpoint and final capture after
      a normal/stopped run; no asynchronous snapshot of mutating stores.
- [x] Preserve the last committed capture if the next generation is interrupted.
- [x] Keep branch/source exports path-allowlisted and tied to exact Git commits.

The exporter binds every revision to raw Git commit objects, reconstructed trees,
ancestry, source partitions, and exact evaluator fingerprints. Complete runs use
their terminal frontier; stopped and crashed runs use only the last jointly durable
checkpoint prefix. `CURRENT` is a lineage-local marker: stable run identity must
match exactly, while a changed capture-configuration fingerprint creates a valid
successor generation rather than a second run.

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

- [x] Exclude `.env`, credentials, Git credential/config material, VCS history,
      caches, datasets, model weights, hidden evaluator material, raw task output,
      and unapproved logs before durable publication.
- [x] Validate paths before copy; reject traversal, device
      files, unexpected symlinks, and submodules.
- [x] Apply exact scope/task-family secret, identity, contamination, license,
      dependency, and artifact-class policies.
- [x] Retain safe structured observations and content-addressed allowlisted source
      artifacts only.
- [x] Record scanner/policy versions, findings, excluded paths/classes, and taint
      sources.
- [ ] An optional coding-agent semantic sweep may only escalate surviving content
      for review; deterministic rejection cannot be overridden. This optional
      enhancement is intentionally not required by M3 and is not implemented.
- [x] Delete raw quarantine according to configured retention only after bundle
      publication/verification or explicit rejection recording.

Free-form model feedback, technical difficulties, raw node output, and raw
ideation calls are projected out. Safe outcome state and evaluator-declared
measurements remain mutually consistent across the journal and experiment-history
projection. The complete sanitation policy contributes to a content fingerprint.
Descriptor-pinned filesystem operations reject symlink/inode replacement during
write, verification, and cleanup; a rejected generation retains a restricted
report without advancing the accepted bundle.

## Bundle storage

`RunBundlePublisher`:

- [x] Builds the exact M1 `RunBundle` manifest from sanitized content.
- [x] Content-addresses referenced blobs and avoids duplicate manifest payloads.
- [x] Preserves completion state and the honest capture frontier.
- [x] Publishes a later capture as a superseding bundle, never an overwrite.
- [x] Commits the bundle to an atomic local content-addressed store consumed by
      M4; it never performs GitHub calls itself.
- [x] Stores one canonical manifest at
      `bundles/<bundle-id-sha256-hex>/manifest.json` and each referenced byte
      payload once at `objects/sha256/<blob-sha256-hex>`; no second refs map
      duplicates the manifest checksum authority.
- [x] Keeps `runs/<run-id-sha256-hex>/current.json` as mutable publisher-only
      control state. It is excluded from the exact reader and from any replicated
      immutable byte authority.
- [x] Serializes cooperating publishers on the pinned store directory, uses
      durable atomic rename boundaries, and safely reclaims one fixed staging
      path per target after interruption.
- [x] Acquires the exporter lease for quarantine retention, verifies the export
      marker and its manifest, counts only marker-committed generations, and
      leaves newer crash-recovery directories for the exporter to reconcile.
- [x] Exposes a separate exact-ID, read-only store that never follows the mutable
      run marker, bounds control/object reads before allocation, verifies the
      admitted sanitation closure, and returns one frozen in-memory byte snapshot.
- [x] Performs no remote publication; normalized M5 snapshot records are never
      treated as substitutes for these raw sanitized bytes.
- [ ] M9/M10 production composition must retain the exact sanitized bundle asset
      closure in a durable locator before a task workspace can be pruned; the
      local M3 store is the only implemented byte authority today.
- [x] Does not label outcomes positive/negative, propose claims, or trigger expert
      evolution.

As established by the system threat model, the Kapso OS account is trusted. The
store rejects pre-existing links, special files, corrupt state, and replaced
authoritative directories, while the directory lease coordinates Kapso
publishers. It is not a sandbox against a hostile process running as the same UID;
that deployment requires the separate-UID, root-owned service boundary described
in the main design.

Checkpoint and experiment-history locations are typed `cross_run.capture`
settings, not duplicated literals. M3 implements and injects `RunCapturePipeline`;
M9 remains responsible for composing those global settings into every workload
launch and constructing the pinned capture context. Missing composition fails
loud rather than silently disabling capture.

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
- Round-trip a bundle through its local immutable store and M4 projection input.

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
