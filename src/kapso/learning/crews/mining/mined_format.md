# The mined view — format contract

You are producing `mined/` — a derived OKF bundle: the campaign's story,
reassembled. Markdown with YAML frontmatter; prose-first; one file per concept;
`index.md` per directory listing entries as `- [title](path) — hero line`.

## Layout
mined/
├── index.md            # the campaign: objective, outcome, iterations as hero lines
├── it-N/
│   ├── index.md        # the round: lens in force (+ replan rationale), parent
│   │                   # branch, flows as hero lines, round winner
│   └── flow-M.md       # one file per idea flow
├── strategy.md         # lens history as belief → evidence → re-aim, per entry
├── operations.md       # kills, crashes, harness incidents, voids
└── artifacts.md        # what was built into the shared space, per registry

## The flow document
One file per idea that entered selection (rejected ones included). Frontmatter —
only what code parses:

    flow: it-2/flow-3
    status: judged        # ideated | selected-unbuilt | build-failed | evaluated
                          # | judged | champion  (derived: how far the flow went)
    member: codex         # authoring member; lens is in the iteration index
    node: 6               # when implemented: node id, branch, runs
    branch: generic_exp_6
    runs: [run_0005, run_0006, run_0019]
    score: 0.7135699      # score of record, when one exists
    valid: true           # evaluation validity / integrity verdict
    sources:              # where each section's content was recovered from —
      idea: <ref>         # one ref per section you wrote, into the raw bundle
      selection: <ref>

Body — sections in loop order, each present ONLY as far as the flow went:

    ## Idea            ← verbatim, as authored
    ## Selection       ← outcome + the selector's words about THIS idea
    ## Implementation  ← what was actually built, ending with the DRIFT NOTE:
                          build-vs-idea fidelity — faithful / deviated (what, why)
                          / partial (what was dropped) — every claim ref-grounded
    ## Evaluation      ← the run sequence with scores and selection labels; for
                          multi-run flows, the internal story (what the in-session
                          attempts tried, what banked)
    ## Judgment        ← the judge's verdict, verbatim
    ## Difficulties    ← the implementor's report, verbatim

## Non-negotiable policies
1. VERBATIM: idea, selector reason, judgment, difficulties are authored one-shot
   artifacts — carry them whole. Strip terminal escape codes; that is the only
   permitted transformation. Condensation lives only in index hero lines.
2. NO FABRICATION, EXPLICIT GAPS: when an ingredient is unrecoverable, the
   section says so and points at the best partial context — e.g.
   "## Idea — not recoverable in this bundle (workspace absent). Partial
   context: the selector's description of this candidate [ref]; the lens it
   answered [ref]." An absent section (flow ended earlier) and an unrecoverable
   section (flow went further but the record is gone) are different things;
   never conflate them.
3. DEGENERATE-ARTIFACT DETECTION: an artifact can be present but junk — a
   format-example echo ("your feedback message"), an empty tag, a truncated
   stream. Mark it: "## Judgment — degenerate (judge echoed its format
   placeholder) [ref]; no verdict exists for this flow." Never quote junk as if
   it were judgment.
4. REFS EVERYWHERE: every number and every verbatim carries a ref into the raw
   bundle (path, plus #anchor or a locating quote). The frame re-greps your
   quotes; a quote it cannot find is a rejected document. Ref fragments the
   frame verifies mechanically: `#L<n>` / `#L<n>-<m>` (line anchors) and
   `#"literal snippet"` (re-grepped); any other fragment is checked as
   path-existence only.
5. DEDUPE ECHOES: the log repeats content (re-streamed file reads). Identical
   content counts once; cite the first occurrence.
6. LINEAGE SLICING: `changes.log` in a run snapshot is cumulative over the
   branch lineage. Attribute entries to THIS flow only when their dates/context
   fall inside this flow's session; earlier entries belong to ancestor flows.
7. WRITE ONLY inside `mined/`. The raw bundle is read-only (the frame verifies
   this against the manifest hashes).

## Appendix — recovery channels (observed on a wave-4 bundle; places to look, never guarantees)

| Flow ingredient | Recovery channels observed (preference order) | Pitfalls observed |
|---|---|---|
| Idea text | workspace checkpoint / experiment store (absent in that bundle) → `<solution>` blocks streamed in the campaign log during ideation → MCP experiment-history renders streamed in LATER iterations' ideation | streamed lines carry ANSI codes; prompt-echoed format examples must not be mistaken for ideas |
| Selection | selector reasoning streamed in the log (full, with per-candidate verdicts) → `ideation/iter<N>/selector/` workspace artifacts in newer bundles | candidate↔flow identity needs care |
| What was built | per-run code snapshots (`runs/run_NNNN/code/` — includes `PLAN.md`, `MEASUREMENT_PROFILE.md`, `eval_profile.md`, `changes.log`, sometimes `full_evaluation_output.log`) → lane-0 implementation stream in the log | **`changes.log` is cumulative per lineage** — a run's copy contains its parent lane's whole story; slice by date/branch context |
| Evaluation | `runs/run_NNNN/manifest.txt` (score, session, evaluator id) + `private/selection.json` (final/superseded/self-voided, with void reasons) + `private/metrics.json` | one flow can hold many runs (nine observed); the session field is the run↔flow join key |
| Judgment | judge feedback printed in the log per iteration result; richest verdicts contain banking directives, economics grades, axis frontiers | **feedback can be the literal placeholder "your feedback message"** (judge echoed its format example) — present-but-degenerate, and the junk propagates into later renders |
| Difficulties | node XML in checkpoint/store (absent in that bundle) → dated `difficulty:` entries inside `changes.log` snapshots → difficulties echoed in MCP renders | same lineage-slicing and echo-duplication cautions |
| Lens/strategy | `lens_plan_history.jsonl` (absent in that bundle) → full lens texts and revision rationales streamed in the log | revision rationales quote judge content that may itself be degenerate |
| Ops events | session end-fact warnings in the log (deadline kills with durations), void records, harness warnings | lane-0-only streaming: other lanes' sessions are visible only through their committed artifacts |

Cross-cutting: the campaign log duplicates some content ~20× (session re-reads
of the same file get re-streamed) — dedupe by content, count once. Bundles vary
by evolve version and completeness: every channel above is a place to look,
never a guarantee.
