# File: `utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 82 |
| Functions | `download_gpt2_files`, `load_gpt2_params_from_tf_ckpt`, `load_encoder_hparams_and_params` |
| Imports | encoder, json, numpy, os, re, requests, tensorflow, tqdm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model loading utilities - downloads and parses GPT-2 weights.

**Mechanism:** Provides three key functions: (1) `download_gpt2_files()` - downloads official GPT-2 model files (checkpoint, encoder.json, hparams.json, vocab.bpe) from OpenAI's public Azure blob storage, supporting all 4 model sizes (124M, 355M, 774M, 1558M). (2) `load_gpt2_params_from_tf_ckpt()` - reads TensorFlow checkpoint files and restructures the weights into a nested dict format expected by the model, organizing transformer block parameters by layer number. (3) `load_encoder_hparams_and_params()` - main entry point that orchestrates downloading (if needed), encoder initialization, hyperparameter loading, and weight extraction.

**Significance:** Essential infrastructure - handles the complexity of loading pre-trained weights so the GPT-2 implementations can focus purely on the forward pass logic. Called by both `gpt2.py` and `gpt2_pico.py` to get model parameters.
