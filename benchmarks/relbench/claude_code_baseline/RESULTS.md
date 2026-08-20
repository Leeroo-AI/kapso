# Claude Code baseline — results

| Task | Fam | Metric | Claude Code | Kapso | Best KumoRFM |
|---|---|---|---|---|---|
| rel-trial/study-outcome | clf | AUROC | 75.16 | 82.1 | 72.0 (v2) |
| rel-amazon/user-churn | clf | AUROC | 71.26 | 71.6 | 70.5 (ft) |
| rel-trial/study-adverse | reg | NMAE | 0.0913 | 0.0872 | 0.128 (v2) |
| rel-amazon/item-ltv | reg | NMAE | 0.0696 | 0.0655 | 0.0795 (v2) |
| rel-trial/site-sponsor-run | rec | MAP | 20.84 | 33.3 | 28.0 (ft) |
| rel-avito/user-ad-visit | rec | MAP | 4.17 | 4.20 | 4.17 (ft) |

Dropped mid-study (baseline outperformed Kapso; killed/replaced per
protocol): rel-stack/post-post-related (val MAP 26.3 vs 19.9),
rel-amazon/user-item-review (val MAP 4.49 vs 3.81),
rel-trial/condition-sponsor-run (test MAP 12.59 vs 12.28),
rel-hm/user-item-purchase (test MAP 3.32 vs 3.26),
rel-amazon/user-item-rate (val MAP 4.11 vs 2.68).
Run artifacts: gs://leeroo-kapso-relbench-artifacts/baselines/claude_code/.
