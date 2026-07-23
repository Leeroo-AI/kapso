# Learning-extraction agent brief (per completed run)

You are mining ONE completed kapso run on a related IOAI task for lessons
**transferable to the target task: IOAI-2026 "Night Watch"**.

## The target task (what "transfer" means)
Night Watch: extend a FROZEN 16-class Audio Spectrogram Transformer to 13 new
classes without forgetting; tiny imbalanced data (3–62 rows/class); ~10-min
single-GPU training; scored 0.5·Acc_old + 0.5·Acc_new.

## What Night Watch ALREADY knows — tag against this, do NOT re-report as NEW
Six prior campaigns established the following. A transfer lesson is **CONFIRMS**
if it restates any of these, and **NEW** only if it does not:
1. The winning recipe is FROZEN-FEATURE: frozen encoder + frozen old head +
   prototype/router for new classes + one metric-tuned bias. Ceiling = 0.86382.
2. Fine-tuning the encoder COLLAPSES old-class accuracy (session-memorization
   on tiny data) — same failure mode as retraining the old head rows.
3. Random / per-sample CV LEAKS (session features) and inflates ~13pp; grouped /
   held-out-by-source splits are the honest, conservative proxy.
4. The residual deficit is SESSION-GENERALIZATION failure baked into the frozen
   features (fine on random splits, collapses on unseen sessions) — invisible to
   random-OOF; unfixable by head, detector, or fine-tune.
5. Macro / balanced scoring ⇒ optimize the old-vs-new balance, not raw accuracy.

## Inputs (read ALL — GREP the big files, never read them whole)
- `run.log` / `run_*.log` — grep for: `member .*candidates`, `<solution`,
  `Node expansion`, `Round winner`, `SHIP=`, `best_score`, `Feedback`,
  `acc_old|acc_new|Macro-F1|OOF`, `\[thinking\]`.
- `results.json`, `task/best_score.log` — final + banked scores.
- `task/kapso_campaign/experiment_history.json` — node history.
- `task/kapso_campaign/.kapso/lens_plan.json` — the lenses.
- `task/kapso_campaign/sessions/*/` — `PLAN.md`, `changes.log`, `result.json`,
  `eval_profile.md`.
- `task/submission/solution.py` — the shipped solution.

## Output — write ONLY this JSON (nothing else) to the path the runner gives
```json
{
  "task": "<name>",
  "final_score": <float or null>, "metric": "<name>",
  "what_worked": [
    {"technique": "...", "evidence": "score delta on which split / the log line"}
  ],
  "what_failed": [
    {"technique": "...", "why": "the failure MECHANISM (often the real payload)",
     "evidence": "..."}
  ],
  "validation_lessons": [
    "split design / leakage caught / proxy-vs-truth gap / noise floor — with evidence"
  ],
  "transfer_to_night_watch": [
    {"lesson": "concrete", "tag": "NEW|CONFIRMS",
     "action": "what a Night Watch run should DO differently",
     "source": "which artifact"}
  ],
  "novelty": "high|medium|low"
}
```

Rules: cite the specific artifact/line for every claim; never invent results the
logs don't show; if the run failed or produced little, report that honestly. A
**low-novelty / all-CONFIRMS** result is valid and useful — do NOT manufacture
NEW lessons to seem productive.
