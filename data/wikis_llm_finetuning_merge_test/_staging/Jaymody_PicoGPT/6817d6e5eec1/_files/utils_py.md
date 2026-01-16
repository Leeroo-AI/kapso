# File: `utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 82 |
| Functions | `download_gpt2_files`, `load_gpt2_params_from_tf_ckpt`, `load_encoder_hparams_and_params` |
| Imports | encoder, json, numpy, os, re, requests, tensorflow, tqdm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model loading and weight downloading utilities.

**Mechanism:** Three main functions: `download_gpt2_files()` fetches official OpenAI GPT-2 checkpoint files (encoder.json, vocab.bpe, model.ckpt.*) from Azure blob storage with progress bars. `load_gpt2_params_from_tf_ckpt()` converts TensorFlow checkpoint format to a nested dict structure compatible with the NumPy implementation - parses variable names like "model/h0/attn/c_attn/w" into nested keys and extracts weight arrays. `load_encoder_hparams_and_params()` is the main entry point that orchestrates: checks for existing checkpoint, downloads if needed, loads the BPE encoder via `encoder.py`, loads hyperparameters from hparams.json, and loads model weights.

**Significance:** Critical infrastructure - enables using official pre-trained GPT-2 weights. Handles the complexity of downloading ~500MB-6GB model files and converting TensorFlow checkpoints to NumPy arrays. Supports all model sizes: 124M, 355M, 774M, 1558M.
