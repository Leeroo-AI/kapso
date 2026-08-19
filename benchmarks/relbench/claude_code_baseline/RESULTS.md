# Claude Code baseline — results

One headless Claude Code session per task (`claude-fable-5`, `--effort xhigh`),
4h wall-clock, 1×A100, generic `PROMPT.md`, same sandbox/contract/scoring as
Kapso (see `README.md`). Test scored one-way after the session. Kapso and
best-published columns from `../RESULTS.md` / `../LEADERBOARD_V1.md`.

| Task | Fam | Metric | Claude Code | Kapso | Best published | Run |
|---|---|---|---|---|---|---|
| rel-trial/study-outcome | clf | AUROC | **75.16** (val 71.87; self-terminated at 81 min) | 82.1 | 72.0 (KumoRFM-2) | `20260819T135603` |
| rel-event/user-repeat | clf | AUROC | — | 81.2 | 81.7 (KumoRFM-2) | pending |
| rel-trial/study-adverse | reg | NMAE | — | 0.0872 | 0.1277 (KumoRFM-2) | pending |
| rel-event/user-attendance | reg | NMAE | — | 0.315 | 0.3071 (KumoRFM-2) | pending |
| rel-trial/site-sponsor-run | rec | MAP | — | 33.3 | 19.0 (ID-GNN-4L) | pending |
| rel-stack/user-post-comment | rec | MAP | — | 13.1 | 13.8 (ID-GNN-4L) | pending |
