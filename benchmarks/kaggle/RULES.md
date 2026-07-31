# Competition rules — binding on every submission

Set by the IOAI AI Models Track organizers. They apply to every kernel you
submit, on every task. Breaking one can void the submission.

**This file overrides the task statement and the competition pages wherever they
disagree.** Those pages carry stale text the organizers have since corrected.

## Submissions

**You get 50 submissions per task — never 5.** Competition pages may still quote
a five-per-day cap; it is obsolete and does not apply. Use the budget: shipping
several genuinely different attempts beats polishing one.

## The kernel

1. **Use ONE GPU.** `machine_shape: NvidiaTeslaT4` provisions **two** T4s; the
   organizers require you to use only one. Pin `device = "cuda:0"` and never
   allocate on `cuda:1` (no `DataParallel`, no `device_map="auto"`). This keeps
   parity with human contestants, who get a single ~18 GB slice.

2. **CPU or T4 only — never P100.**
   - CPU: `"enable_gpu": "false"`, `"machine_shape": ""`
   - GPU: `"enable_gpu": "true"`, `"machine_shape": "NvidiaTeslaT4"`

   P100 can be provisioned but cannot train (known Kaggle bug, will not be
   fixed) and still burns GPU quota. TPU is unsupported for code competitions.

3. **No internet.** The kernel runs with networking disabled — anything that
   downloads at runtime fails.

4. **No outside models or data.** You may use only:
   - models and data shipped **inside** an installed package, and
   - the files supplied with the task (including any provided checkpoint).

   You may **not** download pretrained weights, attach datasets or kernel
   outputs you produced yourself, or use external/scraped/API-sourced data.
   Train inside the kernel from the task's own data.

5. **Submit a `.py` script only** — no trained checkpoints. Every model must be
   trained on the same standard Kaggle hardware.

## Packages

**Your own machine: unrestricted.** Install anything, experiment on any hardware
you have. Only the kernel is constrained.

**The kernel: a fixed environment you cannot add to.** Source of truth —
<https://github.com/IOAI-official/IOAI-AI-Models-Track> (`kaggle-requirements.txt`,
266 packages). Verify against it before relying on an import.

Key pins: `torch 2.13.0` · `torchvision 0.28.0` · `torchaudio 2.11.0` ·
`transformers 5.14.1` · `numpy 2.4.6` · `pandas 3.0.5` · `scikit-learn 1.9.0` ·
`scipy 1.18.0` · `accelerate 1.14.0` · `peft 0.19.1` · `datasets 5.0.0` ·
`safetensors 0.8.0` · `tokenizers 0.22.2` · `librosa 0.11.0` ·
`soundfile 0.14.0` · `opencv-python 5.0.0.93` · `pillow 12.3.0` ·
`matplotlib 3.11.1` · `lightgbm 4.7.0` · `xgboost 3.3.0` · `catboost 1.2.10` ·
`polars 1.43.0` · `networkx 3.6.1` · `nltk 3.10.0` · `spacy 3.8.14` ·
`gensim 4.4.0` · `torchmetrics 1.9.0` · `evaluate 0.4.6`

**Absent — do not import:** `timm`, `einops`, `sentencepiece`, `optuna`,
`statsmodels`, `gymnasium`, `jax`, `tensorflow`, `keras`.
