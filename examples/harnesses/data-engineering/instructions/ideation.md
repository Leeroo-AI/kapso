# Ideation

Propose causes to test, not edits to try. Each idea names the mechanism, where in the
lineage it sits, the query that reveals it, and the check that will prove the fix.

Exploration axes, statused every iteration as active, saturated or deferred:

1. what changed upstream: a source, a producer release, a schema, a volume, a backfill;
   the first question on every ticket (lineage-bisection)
2. the mechanism (data-bug-mechanisms): cardinality, null semantics, status, time
   semantics, ordering and frames, or the data read wrong (a cumulative summed, a code
   misunderstood)
3. the location: the dataset, a parent, a shared intermediate, the join between two
   sources, the load or the orchestration around it
4. the proof: which check fails now and will pass, which downstream datasets may change,
   and for a refactor or a migration the keyed comparison that shows equivalence
5. for a reliability ticket, the pipeline-reliability axes (one logical effect per retry,
   incremental boundaries, late data, backfills); for a cost or speed ticket, the
   warehouse-cost axes (what is billed, what is re-read, what is idle, what is unread)

- Every ticket in a plan carries its proof steps, because the session that builds
  follows the plan: the check that fails now (on what production holds, or on the code
  as it stands) and will pass, the baseline before the change, the proof run after it,
  and what reads the changed dataset. A plan of fixes alone yields unmeasured passes.
- Look at what production holds before proposing a cause (the reference target reads it):
  the deployed table next to the code's output often shows which of the two is wrong.
- With a backlog (ticket-backlog) an idea is a batch: the tickets it takes in lineage
  order, each with its mechanism or its acceptance statement, sized to what one session
  can close with proofs; measurements the feedback says a closed ticket still owes come
  first, as a batch of their own. Start from the ledger of the attempt it builds on
  (results/backlog.json): what is closed stays closed, and a parked ticket's note is
  where its next idea begins. With no ledger yet, every ticket of the inventory
  (prepared/backlog/inventory.json) is open, and its baseline is taken when it is worked.
- A ticket often holds several bugs: one idea per bug, upstream first; the symptom moves
  after each. An idea that changes the dataset until a number matches the ticket is a
  symptom patch; reject it before it is built.
- With one ticket, read the prepared spec (prepared/baseline) and the reproduced number in
  the domain statement: the first idea explains the gap between them and names the check
  that will fail on the code as it stands. A premise the reproduction could not confirm
  is an idea too: show it wrong.
