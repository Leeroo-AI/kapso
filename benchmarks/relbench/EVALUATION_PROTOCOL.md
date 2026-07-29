# RelBench evaluation protocol — per-task temporal regimes

Single authority on **what data a solution may see at test time**, per task. Written
2026-07-29 after the driver-position campaign discovered that the leaderboard mixes two
temporal regimes on long-window tasks.

**Ruling (user decision, 2026-07-29):** where RelBench's library default and the published
bar-setters' released evaluation (KumoRFM) disagree, **this campaign adopts the KumoRFM
regime** — its numbers are the comparison target on nearly every cell we chase.

## The two regimes

Both regimes forbid using any data dated after the row's own seed time. They differ in what
the database contains:

- **frozen-db** — the database handed to the model is truncated at
  `dataset.test_timestamp`; every test row, even one dated years later, is answered from
  that frozen snapshot. This is relbench's `get_db()` default and what its reference
  baselines do.
- **seed-time** — the database is full; each test row is anchored at its own timestamp
  and may use everything strictly before it (rolling-forecast semantics). This is what
  KumoRFM's released benchmark scripts do.

For a task whose test seed times all sit at `test_timestamp` (span 0), the two regimes are
**identical**. They diverge only when the test table extends past `test_timestamp`.

## Primary-source evidence

- RelBench paper (arXiv:2407.20060 §2), verbatim: *"The seed time indicates at which time
  the target is to be predicted, filtering future data."* and *"Our implementation
  carefully hides data after test_timestamp during inference to systematically avoid test
  time data leakage."*
- relbench library: `Dataset.get_db()` truncates at `test_timestamp` by default
  (docstring: "to prevent test leakage"); README calls it a guard against "accidental
  temporal leakage" and shows `get_db(upto_test_timestamp=False)` for the full db.
- RelBench reference baselines (`examples/gnn_entity.py`, `lightgbm_entity.py`,
  `baseline_entity.py`): all plain `get_db()` → frozen-db.
- KumoRFM replication suite (github.com/kumo-ai/kumo-rfm, `benchmarks/v1/relbench/`,
  README: "replicate the benchmarking results on RelBench"):
  `relbench_regression.py:60` and `relbench_classification.py:64` both
  `get_db(upto_test_timestamp=False)`; prediction via `anchor_time='entity'` (per-row
  seed-time anchors); test targets explicitly nulled in context ("Do not leak test
  labels"). KumoRFM-2 paper (arXiv:2604.12596): subgraphs G≤t_i[v_i] "up to timestamp
  t_i". Their v2 scripts and `salt.py` use the same per-row anchoring.
- Board clustering confirms the regimes empirically on rel-f1/driver-position (test MAE):
  frozen cluster — LightGBM 4.170, RDL GraphSAGE 4.022, RelAgent 4.019, Kapso 3.531;
  seed-time cluster — KumoRFM-ft 2.731, KumoRFM-IC 2.747, KumoRFM-2-IC 2.854.

## Sensitivity rule

- **Windowed tasks** (entity clf/reg + recommendation): test seed times =
  `test_timestamp + k·timedelta`, k = 0…`num_eval_timestamps`−1, so
  **span = (num_eval_timestamps − 1) × timedelta**. Span 0 ⇒ regimes coincide ⇒ any
  published bar is comparable and the existing sandbox is already correct.
- **Autocomplete tasks**: test rows are source-table rows after `test_timestamp` (span =
  their time extent). Both KumoRFM (`salt.py`: full timeline, target column of the whole
  test partition set to None, `anchor_time='entity'`) and our sandbox (docstring:
  database keeps post-cutoff rows; target-column values after the cutoff blanked)
  implement the **same semantics** — the family is never regime-divergent, spans are
  listed for context only.

## Verdicts

1. **40 windowed tasks, span 0** — regimes coincide. Nothing to do. (Includes our
   recorded `rel-event/user-attendance` cell — regime-clean.)
2. **3 rel-f1 windowed tasks — the only sensitive cells**: `driver-position` (40×60d =
   6.4y), `driver-dnf` (40×30d = 3.2y), `driver-top3` (40×30d = 3.2y). Bars are
   seed-time (KumoRFM); the sandbox physically enforces frozen-db. **Do not campaign
   these until the sandbox implements per-row censoring.** Existing Kapso driver-position
   cells were recorded under frozen-db and undersell the seed-time score.
3. **23 autocomplete tasks** — sandbox already matches KumoRFM semantics; no gate.
   (relbench's `make_table` for `rel-event/event_interest-*` OOMs >40 GB on a 61 GB box;
   their spans below are derived from the source table directly.)
4. **rel-mimic/patient-iculengthofstay** — credential-blocked; geometry 1×1d, span 0.

## Empirical validation (driver-position, 2026-07-29)

Same features, hyperparameters and honest per-row censoring, evaluated in both regimes
(LightGBM, frozen-origin episode training; scripts `myf1.py` / `fresh_eval.py` /
`rolling_eval.py`, session scratchpad; per-seed prediction vectors saved):

| variant | regime | test MAE | NMAE |
|---|---|---|---|
| campaign best (0728h) | frozen-db | 3.5308 | 0.5026 |
| B-lite (≤2009 model + per-tick fresh features) | seed-time | 3.0178 | 0.4296 |
| B-rolling (retrain per tick on closed windows, 3 seeds) | seed-time | 2.653 ± 0.015 | 0.3776 |
| KumoRFM-ft (bar) | seed-time | 2.731 | 0.3887 |

The 0.88 MAE regime effect dwarfs every modelling intervention tried in the frozen
regime (three independent designs converged at 3.53–3.55).

## Per-task table

Span = how far the test table extends past `test_timestamp` (the regime-divergence
window). ⚠ = regime-sensitive (frozen-db vs seed-time give materially different scores).

| Task | Fam | Ver | Test geometry | Span past cutoff | ⚠ | Sandbox vs declared recipe |
|---|---|---|---|---|---|---|
| rel-amazon/user-churn | clf | v1 | 1×91d | 0 | — | regimes coincide |
| rel-amazon/user-ltv | reg | v1 | 1×91d | 0 | — | regimes coincide |
| rel-amazon/item-churn | clf | v1 | 1×91d | 0 | — | regimes coincide |
| rel-amazon/item-ltv | reg | v1 | 1×91d | 0 | — | regimes coincide |
| rel-amazon/user-item-purchase | rec | v1 | 1×91d | 0 | — | regimes coincide |
| rel-amazon/user-item-rate | rec | v1 | 1×91d | 0 | — | regimes coincide |
| rel-amazon/user-item-review | rec | v1 | 1×91d | 0 | — | regimes coincide |
| rel-amazon/review-rating | ac | v2 | autocomplete | 2.7y | — | aligned (timeline kept, target col blanked) |
| rel-avito/ad-ctr | reg | v1 | 1×4d | 0 | — | regimes coincide |
| rel-avito/user-visits | clf | v1 | 1×4d | 0 | — | regimes coincide |
| rel-avito/user-clicks | clf | v1 | 1×4d | 0 | — | regimes coincide |
| rel-avito/user-ad-visit | rec | v1 | 1×4d | 0 | — | regimes coincide |
| rel-avito/searchstream-click | ac | v2 | autocomplete | 6d | — | aligned (timeline kept, target col blanked) |
| rel-avito/searchinfo-isuserloggedon | ac | v2 | autocomplete | 6d | — | aligned (timeline kept, target col blanked) |
| rel-event/user-attendance | reg | v1 | 1×7d | 0 | — | regimes coincide |
| rel-event/user-repeat | clf | v1 | 1×7d | 0 | — | regimes coincide |
| rel-event/user-ignore | clf | v1 | 1×7d | 0 | — | regimes coincide |
| rel-event/event_interest-interested | ac | v2 | autocomplete | 13d | — | aligned (timeline kept, target col blanked) |
| rel-event/event_interest-not_interested | ac | v2 | autocomplete | 13d | — | aligned (timeline kept, target col blanked) |
| rel-event/users-birthyear | ac | v2 | autocomplete | 13d | — | aligned (timeline kept, target col blanked) |
| rel-f1/driver-position | reg | v1 | 40×60d | 6.4y | **⚠** | **frozen — rebuild before campaigning** |
| rel-f1/driver-dnf | clf | v1 | 40×30d | 3.2y | **⚠** | **frozen — rebuild before campaigning** |
| rel-f1/driver-top3 | clf | v1 | 40×30d | 3.2y | **⚠** | **frozen — rebuild before campaigning** |
| rel-f1/driver-circuit-compete | rec | v2 | 1×365d | 0 | — | regimes coincide |
| rel-f1/results-position | ac | v2 | autocomplete | 13.6y | — | aligned (timeline kept, target col blanked) |
| rel-f1/qualifying-position | ac | v2 | autocomplete | 13.6y | — | aligned (timeline kept, target col blanked) |
| rel-hm/user-item-purchase | rec | v1 | 1×7d | 0 | — | regimes coincide |
| rel-hm/user-churn | clf | v1 | 1×7d | 0 | — | regimes coincide |
| rel-hm/item-sales | reg | v1 | 1×7d | 0 | — | regimes coincide |
| rel-hm/transactions-price | ac | v2 | autocomplete | 8d | — | aligned (timeline kept, target col blanked) |
| rel-stack/user-engagement | clf | v1 | 1×91d | 0 | — | regimes coincide |
| rel-stack/post-votes | reg | v1 | 1×91d | 0 | — | regimes coincide |
| rel-stack/user-badge | clf | v1 | 1×91d | 0 | — | regimes coincide |
| rel-stack/user-post-comment | rec | v1 | 1×91d | 0 | — | regimes coincide |
| rel-stack/post-post-related | rec | v1 | 1×91d | 0 | — | regimes coincide |
| rel-stack/badges-class | ac | v2 | autocomplete | 2.7y | — | aligned (timeline kept, target col blanked) |
| rel-mimic/patient-iculengthofstay | clf | v2 | 1×1d | 0 | — | regimes coincide |
| rel-trial/study-outcome | clf | v1 | 1×365d | 0 | — | regimes coincide |
| rel-trial/study-adverse | reg | v1 | 1×365d | 0 | — | regimes coincide |
| rel-trial/site-success | reg | v1 | 1×365d | 0 | — | regimes coincide |
| rel-trial/condition-sponsor-run | rec | v1 | 1×365d | 0 | — | regimes coincide |
| rel-trial/site-sponsor-run | rec | v1 | 1×365d | 0 | — | regimes coincide |
| rel-trial/studies-enrollment | ac | v2 | autocomplete | 2.9y | — | aligned (timeline kept, target col blanked) |
| rel-trial/studies-has_dmc | ac | v2 | autocomplete | 2.9y | — | aligned (timeline kept, target col blanked) |
| rel-trial/eligibilities-adult | ac | v2 | autocomplete | 2.9y | — | aligned (timeline kept, target col blanked) |
| rel-trial/eligibilities-child | ac | v2 | autocomplete | 2.9y | — | aligned (timeline kept, target col blanked) |
| rel-arxiv/paper-citation | clf | v2 | 1×182d | 0 | — | regimes coincide |
| rel-arxiv/author-category | clf | v2 | 1×182d | 0 | — | regimes coincide |
| rel-arxiv/author-publication | reg | v2 | 1×182d | 0 | — | regimes coincide |
| rel-arxiv/paper-paper-cocitation | rec | v2 | 1×182d | 0 | — | regimes coincide |
| rel-salt/item-plant | ac | v2 | autocomplete | 0.5y | — | aligned (timeline kept, target col blanked) |
| rel-salt/item-shippoint | ac | v2 | autocomplete | 0.5y | — | aligned (timeline kept, target col blanked) |
| rel-salt/item-incoterms | ac | v2 | autocomplete | 0.5y | — | aligned (timeline kept, target col blanked) |
| rel-salt/sales-office | ac | v2 | autocomplete | 0.5y | — | aligned (timeline kept, target col blanked) |
| rel-salt/sales-group | ac | v2 | autocomplete | 0.5y | — | aligned (timeline kept, target col blanked) |
| rel-salt/sales-payterms | ac | v2 | autocomplete | 0.5y | — | aligned (timeline kept, target col blanked) |
| rel-salt/sales-shipcond | ac | v2 | autocomplete | 0.5y | — | aligned (timeline kept, target col blanked) |
| rel-salt/sales-incoterms | ac | v2 | autocomplete | 0.5y | — | aligned (timeline kept, target col blanked) |
| rel-ratebeer/beer-churn | clf | v2 | 1×90d | 0 | — | regimes coincide |
| rel-ratebeer/user-churn | clf | v2 | 1×90d | 0 | — | regimes coincide |
| rel-ratebeer/brewer-dormant | clf | v2 | 1×365d | 0 | — | regimes coincide |
| rel-ratebeer/user-count | reg | v2 | 1×90d | 0 | — | regimes coincide |
| rel-ratebeer/user-beer-liked | rec | v2 | 1×90d | 0 | — | regimes coincide |
| rel-ratebeer/user-place-liked | rec | v2 | 1×90d | 0 | — | regimes coincide |
| rel-ratebeer/user-beer-favorite | rec | v2 | 1×90d | 0 | — | regimes coincide |
| rel-ratebeer/beer_ratings-total_score | ac | v2 | autocomplete | 5.1y | — | aligned (timeline kept, target col blanked) |

## Reproduction

- Windowed geometry (no downloads): iterate `relbench.tasks.<db>` task classes and read
  `timedelta` / `num_eval_timestamps`; span = (n−1)×Δ.
- Autocomplete spans: `get_task(ds, task).get_table("test").df[time_col]` extent past
  `get_dataset(ds).test_timestamp` (for `event_interest-*`, read the `event_interest`
  db table directly — task generation OOMs).
- Kumo regime: `git clone github.com/kumo-ai/kumo-rfm` →
  `benchmarks/v1/relbench/relbench_regression.py` (loader line 60, anchors line ~149),
  `benchmarks/v2/salt.py` (target masking + `anchor_time='entity'`).
- Regime clusters: `benchmarks/relbench/data/baselines.json` (sources embedded per entry).
