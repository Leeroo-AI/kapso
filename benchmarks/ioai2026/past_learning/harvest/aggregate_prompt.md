# Learning-aggregation agent brief (once, after the extractions exist)

Synthesize the per-task extraction JSONs into a SINGLE updated Night Watch
`LEARNINGS.md` — the seed every future Night Watch run reads.

## Inputs
- The current Night Watch `LEARNINGS.md` (the MISSION block + measured task-1
  findings).
- One extraction JSON per harvested related task (from the extraction agent).
- The GOLD REFERENCE SOLUTION for each task (the contest designers' official
  answer, e.g. `reference_solution.ipynb`) — a first-class input, not just
  context. Read each one and mine it for canonical techniques the designers
  endorsed: how they extend a frozen model to new classes, how they validate,
  what they deliberately avoid. These were provided AS the gold standard, so
  we should learn from them directly — not only from what our own run
  discovered.

## Output — write the FULL updated `LEARNINGS.md` (not a diff) to the given path
1. **Preserve the MISSION block verbatim** at the very top.
2. **Preserve the existing measured task-1 findings** — do not drop what we know.
3. Add ONE new section `## Cross-task learnings (harvested)` holding:
   - Only the `transfer_to_night_watch` bullets tagged **NEW** across all
     extractions. Each: the lesson, the concrete Night Watch **action** it
     implies, and the `(task)` it came from. Dedup near-identical NEW lessons
     into one line.
   - Then ONE compact line — `Confirmed by related tasks: <task> → <existing
     finding>, …` — capturing the CONFIRMS bullets so the confirmation is
     recorded without bloating the seed.
   - Then a short `### From the gold reference solutions` block: the canonical
     techniques the contest designers used that transfer to Night Watch, each
     tagged NEW|CONFIRMS with the `(task gold)` source. Only include what is
     genuinely transferable; a gold solution that just uses our known recipe is
     a CONFIRMS one-liner.
4. Re-rank the merged headroom directions (existing + any NEW) by expected value
   for beating 0.86382, as a short ordered list.
5. End with exactly one line:
   `HARVEST VERDICT: <did cross-task learning surface a real NEW lever, or mainly
   confirm the existing approach? name the single highest-value takeaway.>`

## Rules
- The seed is read by EVERY future run — keep it TIGHT; every line earns its
  place; never state the same lesson twice.
- Single-source is fine: with one extraction, still produce the verdict.
- Never fabricate a lesson no extraction supports. If the harvest is
  all-CONFIRMS, SAY so plainly in the verdict — "the related tasks validate our
  approach; no new lever" is the correct output when that is the truth.
- The result must remain a valid Night Watch seed: an agent reading it should
  come away with the mission, the proven recipe, the closed dead-ends, and any
  genuinely new lever — nothing else.
