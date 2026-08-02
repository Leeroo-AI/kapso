Critical path: four shared-encoder LoRA fits totaling about 27,700 batch steps; target at least 2.7 batch steps/s so transformer work stays below 175 minutes.
Confirmation points: debug shape gate after serialization; 20-step LoRA throughput/OOM check; fold-2018 and fold-2019 checkpoints; Model A artifacts by 2h45.
Freeze time: 3h20 after full-run start, reserving 40 minutes for Model B inference, calibration, prediction validation, and archive scoring.

# Execution plan

1. Persist the measured evaluation/input profile and deterministic three-slot serialization.
2. Complete lexical residual and frozen-encoder debug pipeline with full-shape artifacts.
3. Measure LoRA throughput, run both forward folds, and select the fixed schedule from internal ROC-AUC plus sampling noise.
4. Fit independent train-only Model A and train-plus-validation Model B, apply OOS-only blend/calibration, and run the immutable full evaluator.
