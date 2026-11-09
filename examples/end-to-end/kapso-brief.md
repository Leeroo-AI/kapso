Objective    Optimize the model in train.py to improve accuracy on the
             Spaceship Titanic dataset: feature engineering, model choice
             and hyperparameters. train_model() returns a fitted model and
             predict_with_model() predicts with it; both keep their names
             and signatures, only their bodies change.
Measured by  python evaluate.py --data-dir ./data --seed 0, in the
             repository: accuracy on a 90/10 train/validation split of
             data/train.csv
Baseline     about 0.50 (a DummyClassifier)
Target       0.78 or above
Rules
  - train_model and predict_with_model keep their names and signatures
  - evaluate.py and the split it makes are never modified
  - data/test.csv is for final predictions only, never for fitting or tuning
Budget       minutes=45 experiments=4
Harness      predictive-modelling
