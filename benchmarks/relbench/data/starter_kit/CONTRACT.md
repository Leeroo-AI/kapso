# Evaluation contract

The harness runs your program twice per experiment from the repository root:

    python main.py --debug     # must finish fast; validates the pipeline end-to-end
    python main.py             # full training run

After each run it loads, validates, and officially scores:

    $KAPSO_RUN_DATA_DIR/val_predictions.npy
    $KAPSO_RUN_DATA_DIR/test_predictions.npy

Rules:

1. Both files must exist after EVERY run (debug included), with rows aligned
   positionally to `task.get_table("val")` / `task.get_table("test")`.
2. Shapes/dtypes — binary: float (N,) probabilities in [0,1]; regression: float (N,);
   multiclass: float (N, C) scores; recommendation: int (N, K) ranked distinct
   destination ids in [0, num_dst).
3. Your search score is the official VALIDATION primary metric, computed by the
   harness. Test metrics are computed privately; you never see them.
4. Data access: relbench API with download=False only. The cache is sanitized and
   read-only; test labels are physically absent. Do not probe for them.
5. You may run `python kapso_datasets/check_predictions.py` yourself after writing
   predictions to pre-validate shapes before the harness does.
