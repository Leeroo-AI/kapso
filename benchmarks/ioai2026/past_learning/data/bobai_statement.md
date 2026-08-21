# Help BOBAI — extend a deployed classifier from 5 to 7 classes

*IOAI 2024 (Sofia, Bulgaria), On-Site Round — "More classification in an
unknown language".*

A client's text classifier for an unknown language is **already deployed** as
a frozen 5-class model. Today they need it to handle **2 new classes (5 and
6)** — but redeploying a new model is a multi-day process, so your solution
must be **built entirely around the model already deployed**. For security the
raw text is never released; you get only **768-dim frozen embeddings** (mBERT
pooled features) for every sample.

## What you are given (`dataset/`)
- `base_classifier.pth` — the deployed model: a `torch.nn.Linear(768, 5)`
  (weight [5,768], bias [5]). It is FROZEN — you may build around it but the
  intended design keeps its old-class behavior intact.
- `train_with_labels.pt` — tensor `[N, 1, 769]`: labeled samples, each
  `[768-d embedding, label]`, labels 0–6 (old 0–4, new 5–6), roughly
  balanced. Build your solution on this.
- `dev_with_labels.pt` — a DISJOINT labeled holdout `[M, 1, 769]` for honest
  self-checking (never train on it). The final score is on a *separate*
  private test split, so treat `dev` as your unbiased proxy.
- `eval_dataset.pt` — tensor `[200, 1, 768]`: unlabeled inputs (format sanity
  only).

## The task
Produce a 7-way classifier over the 768-d embeddings. The deployed 5-way
model already separates the old classes well; your job is to add classes 5
and 6 **without degrading the old ones**. The classic safe recipe is a
frozen-feature router: keep the base model's old-class decisions untouched and
add a lightweight new-class detector (e.g. nearest-class-mean / prototype /
kNN over the frozen embeddings) — no external models or data, embeddings only.

## Scoring
**Macro-F1 across all 7 classes** (`sklearn.f1_score(average="macro")`) on a
held-out labeled test split. Macro-averaging weights every class equally, so
the 2 new classes matter as much as the 5 old ones — optimize the balance,
not raw accuracy.

## Deliverable contract
Write `submission/solution.py` defining:

    class Solution:
        def __init__(self, data_dir: str):
            # data_dir holds train-dev_with_labels.pt + base_classifier.pth;
            # build your prototypes / router / whatever here.
        def predict(self, X):
            # X: torch tensor [N, 1, 768]. Return N integer predictions 0..6.

Self-check any time (from a scratch dir) with the command in your operational
context; it prints `Macro-F1:` and the old-vs-new accuracy breakdown. Only the
frozen `base_classifier.pth` and the provided embeddings may enter the
solution — no external models, weights, or data.
