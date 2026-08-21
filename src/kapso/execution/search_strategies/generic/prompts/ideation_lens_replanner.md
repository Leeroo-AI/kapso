# Ideation Lens Replanner (keep or revise)

You previously designed the LENSES for an ideation ensemble attacking the
GOAL below. New campaign evidence has accumulated since. Decide whether the
current lens set still carries the highest credible return toward the GOAL's
target bar — or re-aim it.

## GOAL

{{problem}}

## Ensemble members (lens 1 goes to member 1, and so on)

{{member_roster}}

## Current lens plan (in force since iteration {{plan_iteration}})

{{previous_lenses}}

Sources behind it:
{{previous_sources}}

Prior decision rationale: {{previous_rationale}}

## Campaign state

{{campaign_state}}

## Judge feedback from the most recent iterations (verbatim)

{{recent_feedback}}

## Current champion's solution (verbatim)

{{champion_solution}}

## Shared-cache artifacts already available (optional context)

{{shared_artifacts_brief}}

## Design axes of the solution space

{{design_axes}}

## Axis-coverage contract (anti-freeze)

Campaigns measurably fail by silently freezing an axis once a champion
exists (e.g. the input-representation axis never revisited while every
iteration swaps model mechanisms). Whether you KEEP or REVISE, your paragraph must assign
each axis above exactly one status:
- ACTIVE — a lens moves it this iteration (say which lens);
- SATURATED — cite the measured evidence (an ablation/importance study, or
  the specific gated attempts on that axis that failed);
- DEFERRED — one-line reason plus the concrete condition that reopens it.
A paragraph that leaves an axis unmentioned is malformed. Statuses are
dated claims, not verdicts: re-defend a SATURATED or DEFERRED claim
whenever new evidence contradicts it.

## How to decide (return economics)

- The campaign is rewarded ONLY for closing the gap to the GOAL's bar. A
  lens set that keeps producing champion-parity candidates while the
  champion is far from the bar has exhausted its return — keeping it out of
  caution is a loss, not a safe default.
- KEEP only if you can argue the current lenses still have the highest
  credible return: name the concrete next wins they enable.
- REVISE when the evidence says an angle is exhausted (stalled scores,
  judge-closed families, ceiling arguments in the feedback): replace the
  exhausted lens with the highest-CEILING untried family that has a credible
  path to the bar. State the replaced lens's exhaustion evidence and why the
  new family's ceiling is meaningfully beyond the champion. You may re-aim
  one lens or all of them; unreplaced lenses must be restated verbatim.
- A lens is a BIAS for the members, not a cage; write each as one dense
  line, the way a research lead would re-brief a sub-team. WebSearch is
  available if a new family needs grounding — budget it tightly.

## Output format (STRICT — the system parses these tags)

EITHER, to keep the current plan unchanged:

<keep>one-paragraph return argument for keeping the current lenses</keep>

OR, to revise, the full new plan ({{lens_count}} lenses, member order):

<revision_rationale>what changed and why, incl. exhaustion evidence</revision_rationale>
<lens_1>...</lens_1>
<lens_2>...</lens_2>
<sources>
- ...
</sources>
