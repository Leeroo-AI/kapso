Critical path: causally reconstructed customer-week feature matrix and four-fold OOF predictions; target at least 25 retained numeric features/minute and one completed fold/model every 8 minutes.
Confirmation points: core matrix plus core holdout by T+45 minutes, widened matrix plus four-fold gates by T+150 minutes, final Model A/B artifacts by T+270 minutes.
Freeze time: T+292 minutes, reserving at least 45 minutes for full registered evaluation, artifact validation, and reporting.

# Plan

1. Preserve split row order, build the customer-day/customer-week core, and bank its matrix and forward-holdout predictions.
2. Add causal weekly-state, article, cohort, price, popularity, and global-regime blocks using only observations at or before each seed.
3. Compare core and widened LightGBM plus gated CatBoost/XGBoost on four purged training-only forward folds, with temporal weighting and rounds chosen from those folds.
4. Train Model A on train only and freeze validation predictions; train Model B on train plus validation for test.
5. Validate both arrays, run the immutable full evaluator in the foreground, and preserve its complete manifest-bearing output.
