# Claude Code baseline — results

One headless Claude Code session per task (`claude-fable-5`, `--effort xhigh`),
4h wall-clock, 1×A100, generic `PROMPT.md`, same sandbox/contract/scoring as
Kapso (see `README.md`). Test scored one-way after the session. Kapso and
best-published columns from `../RESULTS.md` / `../LEADERBOARD_V1.md`.

Task selection rule: two per category, picked by the largest
Kapso-over-KumoRFM margin (best KumoRFM variant — ft / v1 / v2 in-context —
as the reference), per the campaign's KumoRFM-focused comparison.

| Task | Fam | Metric | Claude Code | Kapso | Best KumoRFM | Kapso gap | Run |
|---|---|---|---|---|---|---|---|
| rel-trial/study-outcome | clf | AUROC | **75.16** (val 71.87; self-terminated at 81 min) | 82.1 | 72.0 (v2) | +10.1 | `20260819T135603` |
| rel-amazon/user-churn | clf | AUROC | — | 71.6 | 70.5 (ft) | +1.1 | pending |
| rel-trial/study-adverse | reg | NMAE | — | 0.0872 | 0.128 (v2) | 32% better | pending |
| rel-amazon/item-ltv | reg | NMAE | — | 0.0655 | 0.0795 (v2) | 18% better | pending |
| rel-stack/post-post-related | rec | MAP | — | 21.8 (26.1 banked Jul) | 12.2 (ft) | +79% | pending |
| rel-amazon/user-item-review | rec | MAP | — | 2.95 | 1.63 (ft) | +81% | pending |

clf note: after study-outcome the board's clf gaps over KumoRFM are all
narrow; user-churn (+1.1 AUROC) is the largest remaining.
Aborted under earlier selection rules (no results recorded):
rel-event/user-repeat (31 min), rel-hm/user-churn (2 min).
