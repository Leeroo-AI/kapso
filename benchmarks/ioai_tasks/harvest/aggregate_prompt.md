# Learning-aggregation agent brief (once, after all extractions)

You are synthesizing the per-task learning docs (from the extraction agents)
into a SINGLE updated `LEARNINGS.md` for the Night Watch shared-cache seed.

## Inputs
- The current Night Watch `LEARNINGS.md` (the existing seed — includes the
  MISSION block and the measured task-1 findings).
- One extraction JSON per harvested related task (Help_BOBAI, and later
  Weather / Chicken_Counting).

## What to produce
An updated `LEARNINGS.md` that a future Night Watch run will read as its seed.
Rules:
- **Preserve the MISSION block verbatim** at the top (standing 0.86382, aim
  for significant gain, don't re-submit baseline, escape the frozen family).
- Add a new section `## Cross-task learnings (harvested)` holding only the
  bullets tagged `NEW` by the extractors — the genuinely additive signal.
  Drop everything tagged `CONFIRMS` (it is already in the seed); instead add
  ONE line noting which related tasks confirmed which existing findings, so we
  know the confirmation happened without bloating the seed.
- For each NEW lesson, state the concrete action it implies for Night Watch
  and which related task it came from.
- Rank the merged headroom directions by expected value for beating 0.86382.
- Keep it tight — every line must earn its place; a seed that is mostly
  confirmation with little new should SAY that (an honest "the related tasks
  largely confirm our approach; the one new lever is X" is the right output if
  that is what the data shows).
- End with a one-line `HARVEST VERDICT`: did cross-task learning surface a
  real new lever for Night Watch, or mainly validate the existing approach?

Output the full new `LEARNINGS.md` content. Do not fabricate lessons that no
extraction doc supports.
