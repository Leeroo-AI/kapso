# Antique Painting Authentication — semi-supervised binary classification

*IOAI 2025 (Beijing), Individual Contest.*

Classify antique paintings as **authentic (1)** or **replica (−1)** from 5
numerical digital features. Professional authentication is expensive, so only a
tiny fraction of paintings are labeled — **the vast majority are unlabeled**.
You must leverage ALL samples, labeled and unlabeled, exploiting the strong
structural patterns in the feature space (semi-supervised learning).

## Data (`dataset/`)
- `training_set.csv` — 500 rows, columns `feature1..feature5` + `Authenticated`.
  Labels: **1 = authentic, −1 = replica, 0 = UNLABELED**. Only ~4 rows are
  labeled (±1); ~496 are unlabeled (0). This is the crux — propagate structure
  from the unlabeled mass, don't just fit the 4 points.
- `validation_with_labels.csv` — 500 rows, `feature1..5` + `label` (∈ {−1, 1}).
  A fully-labeled validation split — your honest self-check / tuning signal
  (the contest's public board). Never train the final decision *on* it in a way
  that won't generalize; the real score is a SEPARATE private test split.

## Scoring
**Accuracy** of predicted labels (∈ {−1, 1}) vs the gold labels on a held-out
private test split (same distribution as validation).

## Deliverable contract
Write `submission/solution.py`:

    class Solution:
        def __init__(self, data_dir: str):
            # data_dir holds training_set.csv (+ validation_with_labels.csv).
            # Build your semi-supervised model here.
        def predict(self, X):
            # X: numpy array [N, 5] of features. Return N labels, each −1 or 1.

Self-check with the command in your operational context (prints `Accuracy:`).
Competition data only — no external data or models.
