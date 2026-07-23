# Learning-extraction agent brief (per completed run)

You are mining ONE completed kapso run on a related IOAI task for lessons
**transferable to the target task: IOAI-2026 "Night Watch"** — an audio
class-incremental task (extend a frozen 16-class Audio Spectrogram Transformer
to 13 new classes without forgetting; tiny imbalanced data; ~10-min single-GPU
budget; scored 0.5·Acc_old+0.5·Acc_new; hard-parts = catastrophic forgetting
under fine-tuning, session/distribution-shift generalization, CV leakage,
frozen features not separating some classes).

## Inputs (salvaged run root — read ALL of them, do not skim)
- `run.log` — the full execution trace (ideation candidates, selector choice,
  per-lane implementation reasoning, feedback verdicts, self-check scores).
- `results.json` / `best_score.log` — final and banked scores.
- `task/kapso_campaign/experiment_history.json` — node-by-node history.
- `task/kapso_campaign/.kapso/lens_plan.json` — the lenses used.
- Every `task/kapso_campaign/sessions/*/` — `PLAN.md`, `changes.log`,
  `result.json`, `eval_profile.md`, and the produced `solution.py`.
- The final `task/submission/solution.py`.

## What to produce (a strict JSON object, via the StructuredOutput schema)
Judge by TRANSFER VALUE to Night Watch, not by whether this task was won.
- `task`: the task name.
- `final_score` + `metric`: the held-out result.
- `what_worked`: techniques that measurably helped, each with the evidence
  (score delta on which split).
- `what_failed`: techniques that were tried and measurably did NOT help, each
  with the WHY (the mechanism of failure — this is often more valuable than
  the wins).
- `validation_lessons`: anything about honest measurement — split design,
  leakage caught, proxy-vs-truth gaps, noise floors.
- `transfer_to_night_watch`: the payload. Concrete bullets, each tagged
  `NEW` (a lesson Night Watch runs have NOT already established) or
  `CONFIRMS` (reinforces what Night Watch already found — frozen-feature
  router, macro-balance, forgetting-avoidance, session-shift validation). Be
  honest: if this task only re-teaches what we know, say so plainly — a
  low-novelty verdict is a valid, useful result.
- `novelty`: one of `high` / `medium` / `low` — how much genuinely new,
  actionable signal this run adds for Night Watch.

Cite the specific artifacts you read. Do not invent results the logs don't
show; if the run failed or produced little, report that honestly.
