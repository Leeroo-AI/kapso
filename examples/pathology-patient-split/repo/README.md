# Growth-pattern classifier

Classify lung adenocarcinoma tiles into four growth patterns: lepidic,
acinar, papillary and solid. Each row of `kapso_datasets/train.csv` is one
tile: its `tile_id`, the `patient_id` it came from, 32 embedding features
(`emb_00` to `emb_31`) and the `pattern` label. `predict.py` trains on
those rows and writes a probability per pattern for every row of
`kapso_datasets/holdout.csv`:

```
python predict.py --data kapso_datasets --out predictions.csv
```

`predictions.csv` has five columns, `tile_id`, `lepidic`, `acinar`,
`papillary` and `solid`, and each row's four probabilities sum to 1. The
evaluation in `kapso_evaluation/evaluate.py` runs that command and scores
the predictions with macro one-vs-rest ROC AUC against labels it holds;
higher is better. The holdout tiles come from patients who appear nowhere
in the training data.

The current model is a logistic regression on the standardised
embeddings, and it prints a validation score from a random split of the
training tiles. Keep the command and the output format; everything else in
this file is yours to change.
