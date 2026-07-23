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
