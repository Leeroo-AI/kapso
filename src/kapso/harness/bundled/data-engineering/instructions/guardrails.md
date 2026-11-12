# Data-engineering guardrails (every module)

Read the task statement first: the change asked for (a "ticket" on these pages, whether
or not a ticketing system exists) or the backlog of them, the stack (engine, transformation
tool, orchestrator), each one's dataset or pipeline, the evaluator, the budget. Its values win.

## Standing rules

- Verify in data, not in code. A change is believed when the built data proves it; a
  green pipeline, a green test and code that reads right are not evidence.
- Every assumption becomes a checked declaration before work starts: the grain (what one
  row means), what a null means, which clock and time zone, how late data may arrive,
  which schema version, which environment, who pays.
- Fix causes, not symptoms. Ask what changed upstream before what is wrong in the SQL,
  reproduce the stakeholder's number first, and bisect the lineage edge by edge. A change
  that makes a number match the ticket without a mechanism is void.
- The proof matches the kind of change: a correctness fix needs a check that failed before
  and passes after; a refactor or an optimisation needs output equivalence by key; a new
  dataset needs an acceptance oracle independent of its own code; a migration needs the
  data and the consumers' contract preserved. Every changed dataset gets a comparison of
  its downstream set against what production holds, wherever a session can read production.
- A task is deterministic from its pinned inputs, prior state, code and interval: a replay
  keeps one logical effect per input; audit before publish; backfills run production's path.
- Shared layers (sources, staging, contracts, snapshots, seeds, schemas) are decisions with
  a blast radius, and each change carries its consumer impact. A test is never changed to
  make a failure go away; a wrong expectation is corrected with independent evidence for
  the corrected contract, and the proof says so.
- Production is read, never written, by a session. Writes go through the code that owns
  the objects, inside the task's authorisation; a change reaches production by the route
  the repository documents (review, then its release). Personal data never leaves the warehouse.
- For a speed or spend ticket the requested metric and the billed quantity are reported
  separately, under recorded conditions, with the rows shown equivalent.
- Confidence is a list of what was checked, against what, and what was not. The label is
  computed from measurements taken before a ticket's first change and after its last
  (validation-report), never chosen, and "the reference is wrong" is a valid finding.

Where to look, each where the module lists it: grain-and-modelling for the canon;
lineage-bisection and data-bug-mechanisms for where a cause hides; invariants, data-diff,
pipeline-reliability, warehouse-cost and engine-portability for the proofs; validation-report
for the claim; ticket-backlog when the task holds several tickets; delivery is the deployment page's.
