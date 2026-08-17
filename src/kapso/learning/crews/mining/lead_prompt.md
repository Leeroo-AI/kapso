Mine this trajectory bundle into mined/ per the contract in mined-format.md.

Bundle: {{trajectory_id}}   (you are at its root)
Manifest: trajectory.yaml — read it first for identity, outcome, inventory.

## Your process

1. SURVEY. Explore the bundle and build the map: which iterations ran, which
   flows existed per iteration (selected AND rejected at ideation), where each
   flow's ingredients live in THIS bundle. Bundles vary by evolve version and
   completeness — the recovery-channels appendix in mined-format.md lists
   places to look, not guarantees. The most stable identities are node ids,
   branch names, and run directories; anchor the map on them. Note what is
   missing or degenerate as you go — those become stated gaps, not silent
   holes.

2. WRITE THE CAMPAIGN GRAIN yourself: mined/index.md (objective, outcome, the
   campaign's story in brief, iterations as hero lines), strategy.md (each lens
   plan and revision as belief → evidence → re-aim, rationales verbatim),
   operations.md (kills with durations, crashes, voids with reasons, harness
   incidents), artifacts.md (the shared-space registry, per producer).
   Write each it-N/index.md shell: lens in force, parent branch, the flow
   roster with one hero line each, round winner — plus the map entries the
   flow writers will need.

3. DECIDE YOUR FAN-OUT. Small campaign (a handful of flows): write the flow
   documents yourself. Larger: delegate via the flow-writer agent — one task
   per iteration by default, per-flow for oversized iterations. You may issue
   several delegations in one message to run them concurrently, but every
   delegation is FOREGROUND: wait for every flow-writer's report before you
   write anything else. NEVER background a delegation — this session ends the
   moment your turn ends, and an in-flight subagent dies with it (a prior run
   lost 10 flow docs exactly this way). Each delegation carries: the
   iteration, its flow roster, and your map entries for it.

4. CRITIC PASS. When all flows are written, spawn the critic agent over the
   full mined/ tree. Address every finding: fix it yourself or re-delegate;
   disagreeing with a finding is allowed but must be answered in the mining
   report, never ignored.

5. SELF-CHECK before finishing: every node id, run directory, and iteration in
   the bundle is accounted for in exactly one flow / iteration doc — or named
   in the report as explicitly skipped, with the reason. Every index.md lists
   exactly the files present.

## Your final message — the mining report
What was written (counts per kind); the map's shape for this bundle version;
every gap and degenerate artifact found, with refs; critic findings addressed
vs disputed (with your answer); anything you could not account for.
The frame will mechanically verify schema, coverage, quotes, and raw
immutability after you finish — findings come back to you by name.
