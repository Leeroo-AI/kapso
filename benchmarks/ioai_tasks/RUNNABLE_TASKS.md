# IOAI-2025 task runnability (for the cross-task learning harvester)

Which 2025 problems we can actually run locally — data obtainable **and** any
required pretrained model reachable from our boxes (HuggingFace / public URL =
OK; Bohrium-only = blocked). "Transfer" = structural similarity to the target
Night Watch audio class-incremental task (T1 twin … T5 negligible), from the
cross-year similarity analysis.

## ✅ RUNNABLE (data + model obtainable)

| Task | Domain | Data source | Pretrained model | HW | Transfer | Worth running? |
|---|---|---|---|---|---|---|
| **Restroom Icon Matching** | Vision | 448 png + 2 npy **in-repo** (45 MB) | OpenAI CLIP ViT-B/16 (public) | GPU (small) | **T3** | Best transfer of runnable set — few-shot prototype / nearest-neighbor matching + frozen-feature separability |
| **Pixel Efficiency** | Vision | HF `IOAI-official/IOAI-2025-Pixel-test` | CLIP (HF) | GPU (small) | **T3** | Frozen-CLIP zero-shot + OOD "reject" class; frozen-feature-separability lens |
| **Chameleon** | NLP | 5 arrow **in-repo** + HF `JettChenT/ioai-chameleon-*` | all-MiniLM-L6-v2 (HF) | CPU / small | T4 | Adapt-a-pretrained-embedding + few-shot, but retrieval (no class-extension) |
| **Antique Painting Auth.** | Tabular | 8 csv **in-repo** (0.2 MB) | none | CPU | T5 | Only "limited labeled data" overlaps; semi-supervised |
| **Radar** (Individual + At-Home) | Vision | 2800 .pt tiles **in-repo** (~1.4 GB each) | none (from-scratch) | GPU (heavy) | T5 | Only extreme class-imbalance weighting overlaps |
| **Synthetic Speech Detector** (GAITE) | **Audio** | 13.7k .pt **in-repo** (679 MB) | none (from-scratch) | GPU | T4 | Same *domain* (spectrogram→CNN) but from-scratch binary classifier — no CIL core |

## Submission format + compute (VERIFIED from the task notebooks)

Read from each notebook's submission cells — not the platform spec. None of
these are a "re-run your full training notebook on hidden data" kernel like the
2026 Kaggle tasks; almost all upload **output predictions**.

| Task | Compute (from notebook) | Time limit | You submit |
|---|---|---|---|
| **Antique** | **CPU** (SVM on tabular features) | none stated | **Predictions** — `submission.zip` = submissionA.csv + submissionB.csv |
| **Radar** | GPU (CNN segmentation, from-scratch) | none stated | **Predictions** — `submission.zip` = submission_val.csv + submission_test.csv |
| **Restroom** | GPU (CLIP ViT-B/16 encoder) | **timed — exceed ⇒ score 0** (value platform-set, not in notebook) | **Predictions** — `submission.zip` = submission_a.npy + submission_b.npy (timed code re-run) |
| **Pixel** | GPU (CLIP) | **explicit: under 8 min for 698 imgs** | **Predictions** — mask coords, ≤6.25% pixels retained (timed code re-run) |
| **Chameleon** | GPU/CPU (SentenceTransformer FT, epochs=1) | none stated | **Checkpoint + code** — `submission.zip` = submission_model.py + `./model/` (fine-tuned model dir) |
| Speech Detector | GPU (spectrogram CNN) | — | (notebook not fetched — UNVERIFIED) |

- **Output predictions (zip of CSV/NPY):** Antique, Radar, Restroom, Pixel.
- **Checkpoint + inference code:** Chameleon (the only one that ships a model).
- **Exact GPU SKU** is a Bohrium platform parameter, not stated in the
  notebooks — not asserting one. All GPU tasks use single-GPU CUDA; Antique is
  pure CPU.
- **Harness implication:** because scoring is predictions-vs-labels (not a
  kernel re-run), these harness like Help_BOBAI / Animal-Deduction — the agent's
  `solution.py` emits predictions, scored on a carved private held-out. No
  kernel-push infrastructure (unlike the Kaggle harness). For Pixel/Restroom the
  harness can enforce the stated time budget as part of the metric.

## ❌ BLOCKED (data not obtainable)

| Task | Why blocked |
|---|---|
| **Weather** (satellite rain) | Inputs (`dataset.npz`, `model_weights.pth`) are Bohrium-only (`/bohr/train-ma50/v2/`); repo has only the scorer + private labels. **T2, highest transfer** — worth it if the two files are supplied. |
| **Chicken_Counting** | Ships the frozen `base.pth` + scorer but **no images** (external). **T2** (freeze-backbone + head + 10-min budget). |
| **Concepts** | No in-repo data; needs an external black-box LLM guesser. T5. |
| **Word_Segmentation** (GAITE) | No in-repo data. |

## Recommendation for the harvest
- Already running: **Help_BOBAI** (IOAI-2024, T1 twin, CPU) — the frozen-router precedent.
- Best obtainable **2025** second task: **Restroom** (T3) — prototype / frozen-feature-separability, the most on-target of the runnable set.
- **Weather** (T2, distribution-shift — the one genuinely non-redundant lesson)
  and **Chicken** (T2) need their Bohrium data supplied to run.
