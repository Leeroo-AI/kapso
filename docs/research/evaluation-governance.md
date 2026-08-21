# Evaluation governance — versioned selection from archive to final report

Design doc, 2026-08-05. Status: **implemented** (all items user-approved
2026-08-05; commits cfc84e30, 080be449, 3476158a, b0ddbf56, 14d564df). No
legacy paths are kept anywhere (Rule 7).

Two deltas from the design as written, both taken during implementation:

1. **Vendoring by copy, not maintainer placement.** The sandbox helper ships
   checked-in at `benchmarks/relbench/data/generic_eval/kapso_eval_archive.py`
   (byte-identical to the framework master, pinned by test) rather than being
   copied into the tree by maintainer setup — provided trees are then
   self-contained and the helper lands inside the provided-immutability
   baseline automatically, with zero maintainer changes.
2. **Run↔rescore agreement is enforced at the last mile, not at calibration.**
   Calibration runs at fast fidelity and fast runs are never archived, so
   there is nothing to rescore at that point. Enforcement is the
   `select_final` tripwire (every pooled run's archive-time score must equal
   its recomputation bit-for-bit) plus a real-grader, real-data agreement
   test; the prompts carry the shared-scoring-path requirement.

## 1. Why

Two live incidents, one root shape:

- **rel-event/user-ignore** — the official validation window is a single
  irregular time slice; candidate ranking under it anti-correlates with test
  (−0.71, selection regret +11.1). The agent had no channel it knew of to get
  the measurement changed, and the last mile shipped the argmax of the broken
  ruler (79.5 over an archived 90.05).
- **rel-f1/driver-top3** — a contract-violating candidate scored highest on
  validation *because of* its defect. In-loop governance (integrity checks,
  comparability classes, evaluator transitions) exists and is tested, but the
  last mile (`final_evaluate`) ignores all of it: it re-scores every final with
  a scorer hardwired to the original protocol.

The framework already owns most of the answer: maintainer-registered
evaluation, `<evaluation_change_request>` routing, accept → re-register →
frontier bridge → `refresh_score_projections` (scores are projections of
append-only attempts; no score crosses an `evaluator_id` boundary; unmeasured
projects `None` and never wins). What is missing:

1. Archives carry **no record of which evaluator produced them**.
2. Final selection has **its own scorer** instead of the registered one.
3. Nothing tells the solving agent **when the ruler itself is the problem**.

## 2. The split

> **Framework owns how measurements are recorded, labeled, versioned, and
> selected. A benchmark owns what a measurement is.**

Everything below follows from that sentence. The archive layout, selection
labels, evaluator stamps, version snapshots, head resolution, and final
selection are generic and move INTO the framework. The relbench grader keeps
only benchmark-specific scoring: task metrics, sanitized-cache access, rolling
tick assembly, boolean-target coercion, test-label custody.

## 3. Components

### 3.1 Framework — new: `kapso.execution.evaluation_archive` **[needs approval]**

Owns the full archive contract, lifted out of the relbench grader:

- **Layout**: `<archive_root>/runs/run_%04d/{*_predictions.npy, private/{metrics.json,
  selection.json}}` plus `<archive_root>/evaluators/<evaluator_id>/` snapshots.
- **Selection labels** (unchanged semantics, relocated): `pending` at archive
  time; `final`/`superseded` stamped via the existing
  `finalize_run_selection` handler hook; `self-voided` via the void channel;
  `invalid` for judge-invalidated runs. Session-final inference (last archived
  run per session) moves here from the relbench handler.
- **Evaluator stamp**: every archive records the fingerprint of the
  `kapso_evaluation/` tree that produced it, written at archive time by
  self-fingerprinting. Per-candidate integrity enforcement already guarantees
  a validly scored candidate's tree matches the registered manifest, so
  stamp == registered head at archive time. A tampered tree stamps an
  off-head id and thereby self-excludes from the final pool.
- **Version snapshot**: first archive under a fingerprint copies the tree to
  `evaluators/<id>/` (idempotent). This is what makes the head's scoring
  logic reachable after the campaign without the strategy workspace.
- **Head resolution**: `head_id = stamp of the highest-numbered archive`.
  Justification: stamps equal the registered head at their archive time, the
  registry only moves forward, and an accepted transition immediately
  produces head-stamped archives via the bridge. When the registry is
  reachable (finalization runs with the workspace in scope), additionally
  assert `head_id == registry.head().evaluator_id` — mismatch raises.
- **Final selection** (`select_final(archive_root, direction)`):
  1. infer session finals; 2. resolve head; 3. verify
  `fingerprint(evaluators/<head>) == head` and that the provided-baseline
  files inside are byte-identical to the registration-time provided manifest;
  4. pool = runs labeled `final` AND stamped head; 5. score each via the
  entrypoint's `--rescore` (below); 6. direction-aware argmax.
  Off-head finals are returned as `excluded` with reasons — reported, never
  ranked (missing evidence never wins, same doctrine as node projections).
  Loud failure on: unstamped archive, missing snapshot, fingerprint mismatch,
  rescore-unsupported tree, empty head pool.
- **Sandbox-vendored helper**: the stamp/snapshot/label/void mechanics ship as
  one self-contained, stdlib+numpy-only module that the maintainer's setup
  places inside `kapso_evaluation/` (it is therefore covered by the manifest
  and immutable to candidates). Benchmark graders import it instead of
  reimplementing archival. Its fingerprint routine is pinned byte-equal to
  `evaluation_integrity.manifest_fingerprint` by a test, since the sandbox
  module cannot import kapso.

### 3.2 Framework — entrypoint contract: `--rescore RUN_DIR` **[needs approval]**

`kapso_eval.py` gains a third mode next to `--fidelity fast|full`:
recompute the `KAPSO_EVAL_MANIFEST` line for an existing archive from its
stored artifacts, executing **no candidate code**. Run and rescore must share
one scoring path. The maintainer prompts (`setup_provided.md`,
`change_request.md`) state the requirement; **calibration mechanically
verifies it** (archive one full run, rescore it, scores must agree) at v1
registration and at every accepted change. An evolved protocol may only
demand artifacts its own run mode makes candidates produce — the version that
requires them is the version that knows how to rescore them.

### 3.3 Framework — teach the agent when to request a change **[needs approval]**

The registered-evaluation instructions shown to every session
(`GenericSearch._evaluation_instructions()`) gain the channel and the
diagnostics. Text (generic predictive-ML, no task specifics):

> **The evaluation protocol is versioned and negotiable — when it mismeasures,
> request a change instead of optimizing the broken measurement.** Two cheap
> diagnostics, neither touching test:
> (a) *Resolution* — bootstrap the validation metric on your best candidate to
> get its standard error, and measure how much your candidates' predictions
> actually differ (mean pairwise rank correlation on validation rows). If
> materially different candidates score within ~2 standard errors of each
> other, validation is not separating them and its argmax is close to a coin
> flip.
> (b) *Representativeness* — if validation is a single time slice, compare its
> event volume and label rate against surrounding history and the prediction
> period. An irregular slice (calendar shock, outage) can rank candidates in
> the WRONG ORDER, not merely with less precision; no tuning fixes an
> inverted ordering.
> If either check fails, do not keep climbing the official score and do not
> silently select on a private alternative. File
> `<evaluation_change_request>` with the measured evidence and a concrete
> proposal — e.g. additional validation windows generated by the task's own
> label-generating code over training-era timestamps, each window closing
> before the prediction period, aggregated so no single slice decides the
> ranking. Requests are triaged against evidence (numbers, not suspicion) and
> the budget is small (3 per campaign): file once, early, with your best
> case. If accepted, everything is re-measured under the new score and prior
> champions may lose rank — that is the system working, not a regression.

The relbench context keeps only benchmark specifics (data rules, prediction
contract, rolling note); the earlier plan to put these diagnostics in
`benchmarks/relbench/context.py` is dropped to avoid duplication.

### 3.4 Benchmark — relbench grader (autonomous)

- Imports the vendored archive helper; **deletes** its bespoke archival,
  labeling, and void plumbing (the behavior is identical, relocated).
- Implements `--rescore` as exactly its current scoring block (load stored
  `val_predictions.npy`, score against pristine labels via the task's own
  metrics, print the manifest line). Rolling archives already store the
  assembled val vector, so rescore is uniform.
- Keeps: sanitized-cache discipline, per-tick rolling assembly,
  boolean-target coercion, test-prediction custody.

### 3.5 Benchmark — relbench handler (autonomous)

`final_evaluate` becomes a thin wrapper over the framework selector:

```
result = evaluation_archive.select_final(self.runs_dir.parent, direction)
# benchmark-specific tail, unchanged in spirit:
#   compute test metrics ONCE for the winner against pristine labels
#   (boolean-coerced, rolling-aware), run the code audit, emit the report
#   with selected_by = "max rescored validation under evaluator <id12> (vN)",
#   head version, excluded finals, voided runs.
```

**Deleted, no legacy** (Rule 7): `_recomputed_val_metrics`,
`_infer_session_finals` (relocated to framework), the hardwired
`task.evaluate` ranking loop. Archives without stamps raise — pre-design
archives are historical artifacts in GCS/RESULTS.md, not inputs to future
selection; nothing re-reads them through this path.

## 4. Flows (condensed)

**Steady state**: candidate → registered command (full) → grader scores,
archives with stamp, snapshots tree if new version, prints manifest →
`node.score` = manifest score; attempt appended under the registered id →
feedback → `finalize_run_selection` labels the session's run-of-record.

**Mid-run change**: candidate emits `<evaluation_change_request>` with the
diagnostics above → maintainer triages (cap 3) → on accept: wrapper rebuilt
(provided bytes immutable), calibration incl. run↔rescore agreement, v2
appended to registry → adopt → bridge re-runs the registered evaluation over
the frontier (requester first, checkpointed) — each success is a fresh
v2-stamped archive → `refresh_score_projections(v2)`: bridged nodes carry
their v2 score, all others project `None` and cannot win → sessions continue
under v2; session finals migrate automatically because the final is the last
archive per session.

**End**: `select_final`: head = newest stamp (cross-checked against the
registry when reachable) → verify snapshot + provided-byte anchor → pool =
head-stamped finals → `--rescore` each → argmax → benchmark tail computes the
winner's test metrics once. A pre-transition final with an inflated old-ruler
score is outside the pool by construction — the certified-broken ruler never
gets the last word.

## 5. Trust model (unchanged or stronger)

Stored scores are never trusted for ranking; ranking values are recomputed
from stored predictions (self-authenticating — forging them just changes your
answers) by a tree that must hash to the stamp, whose provided core is
byte-anchored to the registration-time manifest. New surfaces (stamp,
snapshot) are protected by the same hash: you cannot stamp `head_id` onto a
tree that does not hash to it. All verification failures raise; there are no
fallbacks (Rule 2).

## 6. Tests

Existing (keep): integrity suite, transition suite, fidelity/comparability
suite, `test_evaluation_governance.py`.

New: helper-vs-`manifest_fingerprint` byte-agreement; stamp equals registered
id on a real archive; snapshot idempotence; head resolution (incl. registry
cross-check mismatch → raise); pool filtering with off-head finals reported
and never ranked; run↔rescore agreement on a real small task; unstamped
archive → raise; snapshot tamper → raise; calibration rejects an entrypoint
whose rescore disagrees; instruction text carries no benchmark vocabulary
(pin-test, same style as the context pins).

## 7. Approval checklist

| # | Change | Where | Status |
|---|--------|-------|--------|
| 1 | `evaluation_archive` module + vendored sandbox helper | framework | shipped `cfc84e30` |
| 2 | `--rescore` in entrypoint contract (prompt edits, all three) | framework | shipped `b0ddbf56` |
| 3 | Change-request diagnostics in `_evaluation_instructions()` | framework | shipped `14d564df` |
| 4 | Session-final inference relocation (delete from handler) | framework+benchmark | shipped `cfc84e30`+`3476158a` |
| 5 | Grader: vendored helper, stamp+snapshot, `--rescore` | benchmark | shipped `080be449` |
| 6 | Handler: thin `final_evaluate`, delete legacy scorer | benchmark | shipped `3476158a` |
| 7 | Rolling-contract re-truncation warning | benchmark | shipped `14acf3a6` |

Also landed alongside item 6: `coerce_boolean_target` moved handler →
task_specs, so the sandbox builder's bare subprocess no longer imports the
handler (and, through it, the framework).

## 8. Migration tiers (added after the first live transition)

The first live transition (rel-event/user-ignore, 2026-08-09) accepted a
correct five-window remedy, but the maintainer rebuilt the wrapper
candidate-aware: candidates were asked to produce the extra windows
themselves. Five of the six frontier designs crashed under the new
contract (`exit 5`), the bridge could anchor only one node, and the
12-candidate v1 pool collapsed to a 1-candidate v2 pool at 5h16m of a 6h
cap. Two prompt-side rules now close this, stated generically in terms of
the candidate contract (entrypoint, standard input layout, stored raw
outputs):

- **Maintainer** (`change_request.md`, mirrored in both setup prompts):
  implement the LOWEST migration tier that fixes the defect —
  (1) rescore-only: recompute from outputs runs already store; archived
  runs re-rank via `--rescore` with zero re-execution.
  (2) same-contract re-invocation: prepare each new window/slice/seed in
  the standard layout and invoke the UNCHANGED candidate entrypoint per
  unit, aggregating evaluator-side; prior candidates stay measurable.
  (3) contract-breaking: only when nothing less fixes the defect, declared
  in the verdict reason.
- **Requester** (`_evaluation_instructions()` point 4): run the two
  diagnostics in the first iteration and file at the FIRST confirmation
  (a late transition voids every old-evaluator measurement); propose the
  least-breaking remedy and name its tier; after a transition, porting the
  strongest voided designs to the current contract is called out as the
  highest-value experiment.

No behavior code changed: archives already store raw outputs, `--rescore`
already re-scores them, and the bridge already re-invokes candidates. The
tiers are enforced where the wrapper is authored — in the maintainer's
instructions — and pinned by `test_maintainer_prompts_carry_the_rescore_contract`
and `test_evaluation_instructions_demand_early_filing_and_low_tiers`.
