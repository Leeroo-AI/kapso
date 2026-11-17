Objective    Raise the growth-pattern classifier's macro ROC AUC on new
             patients above 0.88 without changing the prediction contract:
             predict.py keeps writing tile_id and one probability per
             pattern (lepidic, acinar, papillary, solid) for every holdout tile.
Measured by  kapso_evaluation/evaluate.py, macro one-vs-rest ROC AUC over
             the four patterns on the holdout tiles
Baseline     0.842 (the shipped logistic regression, patient-bootstrap SE 0.021)
Target       0.88
Rules
  - the evaluator is never modified
  - no fitting on the holdout labels: they are the evaluator's alone, in the directory `kapso_evaluation/private_dir.txt` names
  - the holdout tiles come from 40 patients who appear nowhere in the training data; `patient_id` names each tile's patient
  - a tile may be read beside the other tiles of its own patient, holdout tiles included (a slide is scored whole); no statistic is pooled across holdout patients
  - the data has no time order: no temporal regime applies
Budget       minutes=45 experiments=3
Harness      predictive-modelling
