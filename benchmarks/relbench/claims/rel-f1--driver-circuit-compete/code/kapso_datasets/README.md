# RelBench starter kit (Kapso)

Known-good building blocks for this task. Adapt them — especially the temporal
sampling / censoring logic, which is easy to get wrong when written from scratch.

| File | What it is |
|---|---|
| `common.py` | Load the task from the sanitized cache (`download=False`), save predictions per the contract, temporal-safety assert, fast MAP@K. Start here. |
| `recommendation_heuristics.py` | Time-decayed repeat-behavior + popularity ranking, co-occurrence candidate generation. The mandatory baseline for recommendation tasks. |
| `vendored_examples/` | Official RelBench reference implementations (MIT), vendored verbatim from the relbench repo. |

Vendored examples map:

- `gnn_entity.py` + `model.py` + `text_embedder.py` — the RDL temporal GNN for entity
  classification/regression (PyTorch Frame encoders + heterogeneous GraphSAGE +
  time-aware NeighborLoader). Supports `--include_task_tables` (past labels as features).
- `idgnn_recommendation.py` — ID-GNN link prediction (the stronger official rec baseline).
- `gnn_recommendation.py` — two-tower GNN variant.
- `gnn_autocomplete.py` — GNN for autocomplete tasks (predict a removed column of a row).
- `lightgbm_entity.py` / `lightgbm_autocomplete.py` / `lightgbm_recommendation.py` — GBDT baselines.
- `baseline_entity.py` — trivial statistical baselines (sanity bars).

Required adaptations when reusing vendored examples:

1. They call `get_dataset(..., download=True)` / `get_task(..., download=True)` — change to
   `download=False` (the sanitized cache is complete; downloads are blocked).
2. They write model caches to `~/.cache/relbench_examples` — point `--cache_dir` (or the
   equivalent variable) at `$KAPSO_SHARED_CACHE_DIR` instead.
3. They only print metrics — add `save_predictions(val_pred, test_pred)` from
   `common.py` so the harness can score the run.
4. Keep row order: predict directly on `task.get_table("val")` / `task.get_table("test")`
   in their given order.
5. Respect `--debug`: subsample training aggressively (max a few thousand seed rows,
   1 epoch) but still write full-shape predictions.
