# Workflow Index: Jaymody_PicoGPT

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Repository Building).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Rough APIs | GitHub URL |
|----------|-------|------------|------------|
| Text_Generation | 7 | `load_encoder_hparams_and_params`, `Encoder.encode`, `generate`, `gpt2`, `Encoder.decode` | PENDING |

---

## Workflow: Jaymody_PicoGPT_Text_Generation

**File:** [→](./workflows/Jaymody_PicoGPT_Text_Generation.md)
**Description:** End-to-end text generation using GPT-2 in pure NumPy.
**GitHub URL:** PENDING

### Steps Overview

| # | Step Name | Rough API | Related Files |
|---|-----------|-----------|---------------|
| 1 | Model Download | `download_gpt2_files` | utils.py |
| 2 | Weight Conversion | `load_gpt2_params_from_tf_ckpt` | utils.py |
| 3 | Tokenizer Initialization | `get_encoder`, `Encoder.__init__` | encoder.py |
| 4 | Input Encoding | `Encoder.encode`, `Encoder.bpe` | encoder.py |
| 5 | Autoregressive Generation | `generate` | gpt2.py |
| 6 | Forward Pass | `gpt2`, `transformer_block`, `mha`, `ffn`, `attention` | gpt2.py |
| 7 | Output Decoding | `Encoder.decode` | encoder.py |

### Source Files (for enrichment)

- `gpt2.py` - Main GPT-2 implementation with `main()`, `generate()`, `gpt2()`, transformer blocks
- `gpt2_pico.py` - Minimal 40-line version of the same (alternative implementation)
- `encoder.py` - BPE tokenizer: `Encoder` class with `encode()`, `decode()`, `bpe()` methods
- `utils.py` - Model loading: `download_gpt2_files()`, `load_gpt2_params_from_tf_ckpt()`, `load_encoder_hparams_and_params()`

---

### Step 1: Model_Download

| Attribute | Value |
|-----------|-------|
| **API Call** | `download_gpt2_files(model_size: str, model_dir: str) -> None` |
| **Source Location** | `utils.py:L13-41` |
| **External Dependencies** | `requests`, `tqdm`, `os` |
| **Key Parameters** | `model_size: str` - GPT-2 model variant ("124M", "355M", "774M", "1558M"), `model_dir: str` - local directory to save checkpoint files |
| **Inputs** | Model size string and target directory path |
| **Outputs** | Downloads checkpoint, encoder.json, vocab.bpe, hparams.json files to model_dir |

---

### Step 2: Weight_Conversion

| Attribute | Value |
|-----------|-------|
| **API Call** | `load_gpt2_params_from_tf_ckpt(tf_ckpt_path: str, hparams: dict) -> dict` |
| **Source Location** | `utils.py:L44-65` |
| **External Dependencies** | `tensorflow`, `numpy`, `re` |
| **Key Parameters** | `tf_ckpt_path: str` - path to TensorFlow checkpoint, `hparams: dict` - hyperparameters dict containing n_layer |
| **Inputs** | TensorFlow checkpoint path and hparams dict |
| **Outputs** | Nested dict `params` with keys: `wte`, `wpe`, `blocks` (list of transformer block weights), `ln_f` |

---

### Step 3: Tokenizer_Initialization

| Attribute | Value |
|-----------|-------|
| **API Call** | `get_encoder(model_name: str, models_dir: str) -> Encoder` |
| **Source Location** | `encoder.py:L114-120` |
| **External Dependencies** | `json`, `os` |
| **Key Parameters** | `model_name: str` - model size identifier ("124M", etc.), `models_dir: str` - directory containing model files |
| **Inputs** | Model name and models directory path |
| **Outputs** | `Encoder` instance initialized with vocabulary and BPE merge rules |

| Attribute | Value |
|-----------|-------|
| **API Call** | `Encoder.__init__(self, encoder: dict, bpe_merges: list, errors: str = "replace") -> None` |
| **Source Location** | `encoder.py:L48-58` |
| **External Dependencies** | `regex` |
| **Key Parameters** | `encoder: dict` - token-to-id mapping, `bpe_merges: list` - list of BPE merge tuples, `errors: str` - error handling mode for decode |
| **Inputs** | Vocabulary dict and BPE merges list |
| **Outputs** | Initialized Encoder with encoder/decoder dicts, byte mappings, BPE ranks, and regex pattern |

---

### Step 4: Input_Encoding

| Attribute | Value |
|-----------|-------|
| **API Call** | `Encoder.encode(self, text: str) -> list[int]` |
| **Source Location** | `encoder.py:L101-106` |
| **External Dependencies** | `regex` |
| **Key Parameters** | `text: str` - input text to tokenize |
| **Inputs** | Raw text string (user prompt) |
| **Outputs** | List of integer token IDs |

| Attribute | Value |
|-----------|-------|
| **API Call** | `Encoder.bpe(self, token: str) -> str` |
| **Source Location** | `encoder.py:L60-99` |
| **External Dependencies** | None (internal method) |
| **Key Parameters** | `token: str` - unicode-encoded token string |
| **Inputs** | Single token string |
| **Outputs** | Space-separated BPE subword string |

---

### Step 5: Autoregressive_Generation

| Attribute | Value |
|-----------|-------|
| **API Call** | `generate(inputs: list[int], params: dict, n_head: int, n_tokens_to_generate: int) -> list[int]` |
| **Source Location** | `gpt2.py:L86-94` |
| **External Dependencies** | `tqdm`, `numpy` |
| **Key Parameters** | `inputs: list[int]` - initial token IDs, `params: dict` - model weights, `n_head: int` - number of attention heads, `n_tokens_to_generate: int` - tokens to generate |
| **Inputs** | Input token IDs, model parameters, number of heads, generation length |
| **Outputs** | List of generated token IDs (only the new tokens, not the prompt) |

---

### Step 6: Forward_Pass

| Attribute | Value |
|-----------|-------|
| **API Call** | `gpt2(inputs: list[int], wte: np.ndarray, wpe: np.ndarray, blocks: list[dict], ln_f: dict, n_head: int) -> np.ndarray` |
| **Source Location** | `gpt2.py:L73-83` |
| **External Dependencies** | `numpy` |
| **Key Parameters** | `inputs: list[int]` - token IDs, `wte: np.ndarray` - token embeddings [n_vocab, n_embd], `wpe: np.ndarray` - positional embeddings [n_ctx, n_embd], `blocks: list[dict]` - transformer block weights, `ln_f: dict` - final layer norm params, `n_head: int` - attention heads |
| **Inputs** | Token IDs and model parameters |
| **Outputs** | Logits array of shape [n_seq, n_vocab] |

| Attribute | Value |
|-----------|-------|
| **API Call** | `transformer_block(x: np.ndarray, mlp: dict, attn: dict, ln_1: dict, ln_2: dict, n_head: int) -> np.ndarray` |
| **Source Location** | `gpt2.py:L63-70` |
| **External Dependencies** | `numpy` |
| **Key Parameters** | `x: np.ndarray` - input [n_seq, n_embd], `mlp: dict` - FFN weights, `attn: dict` - attention weights, `ln_1/ln_2: dict` - layer norm params, `n_head: int` - attention heads |
| **Inputs** | Hidden states and block parameters |
| **Outputs** | Transformed hidden states [n_seq, n_embd] |

| Attribute | Value |
|-----------|-------|
| **API Call** | `mha(x: np.ndarray, c_attn: dict, c_proj: dict, n_head: int) -> np.ndarray` |
| **Source Location** | `gpt2.py:L38-60` |
| **External Dependencies** | `numpy` |
| **Key Parameters** | `x: np.ndarray` - input [n_seq, n_embd], `c_attn: dict` - QKV projection weights, `c_proj: dict` - output projection weights, `n_head: int` - number of attention heads |
| **Inputs** | Normalized hidden states and attention weights |
| **Outputs** | Attention output [n_seq, n_embd] |

| Attribute | Value |
|-----------|-------|
| **API Call** | `ffn(x: np.ndarray, c_fc: dict, c_proj: dict) -> np.ndarray` |
| **Source Location** | `gpt2.py:L24-31` |
| **External Dependencies** | `numpy` |
| **Key Parameters** | `x: np.ndarray` - input [n_seq, n_embd], `c_fc: dict` - expand projection weights (to 4*n_embd), `c_proj: dict` - compress projection weights (back to n_embd) |
| **Inputs** | Normalized hidden states and FFN weights |
| **Outputs** | FFN output [n_seq, n_embd] |

| Attribute | Value |
|-----------|-------|
| **API Call** | `attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray` |
| **Source Location** | `gpt2.py:L34-35` |
| **External Dependencies** | `numpy` |
| **Key Parameters** | `q: np.ndarray` - queries [n_q, d_k], `k: np.ndarray` - keys [n_k, d_k], `v: np.ndarray` - values [n_k, d_v], `mask: np.ndarray` - causal mask [n_q, n_k] |
| **Inputs** | Q, K, V matrices and causal mask |
| **Outputs** | Attention output [n_q, d_v] |

---

### Step 7: Output_Decoding

| Attribute | Value |
|-----------|-------|
| **API Call** | `Encoder.decode(self, tokens: list[int]) -> str` |
| **Source Location** | `encoder.py:L108-111` |
| **External Dependencies** | None |
| **Key Parameters** | `tokens: list[int]` - list of token IDs to decode |
| **Inputs** | List of generated token IDs |
| **Outputs** | Decoded text string |

---

### Implementation Extraction Guide

| Step | API | Source | Dependencies | Type |
|------|-----|--------|--------------|------|
| Model_Download | `download_gpt2_files` | `utils.py:L13-41` | requests, tqdm | API Doc |
| Weight_Conversion | `load_gpt2_params_from_tf_ckpt` | `utils.py:L44-65` | tensorflow, numpy | API Doc |
| Tokenizer_Initialization | `get_encoder` | `encoder.py:L114-120` | json | API Doc |
| Tokenizer_Initialization | `Encoder.__init__` | `encoder.py:L48-58` | regex | API Doc |
| Input_Encoding | `Encoder.encode` | `encoder.py:L101-106` | regex | API Doc |
| Input_Encoding | `Encoder.bpe` | `encoder.py:L60-99` | (internal) | API Doc |
| Autoregressive_Generation | `generate` | `gpt2.py:L86-94` | tqdm, numpy | API Doc |
| Forward_Pass | `gpt2` | `gpt2.py:L73-83` | numpy | API Doc |
| Forward_Pass | `transformer_block` | `gpt2.py:L63-70` | numpy | API Doc |
| Forward_Pass | `mha` | `gpt2.py:L38-60` | numpy | API Doc |
| Forward_Pass | `ffn` | `gpt2.py:L24-31` | numpy | API Doc |
| Forward_Pass | `attention` | `gpt2.py:L34-35` | numpy | API Doc |
| Output_Decoding | `Encoder.decode` | `encoder.py:L108-111` | (none) | API Doc |

---

### Helper Functions Reference

The following utility functions support the main workflow steps:

| Function | Source | Purpose | Type |
|----------|--------|---------|------|
| `bytes_to_unicode` | `encoder.py:L12-32` | UTF-8 byte to unicode mapping for BPE | API Doc |
| `get_pairs` | `encoder.py:L35-44` | Extract symbol pairs from word tuple | API Doc |
| `gelu` | `gpt2.py:L4-5` | GELU activation function | API Doc |
| `softmax` | `gpt2.py:L8-10` | Softmax with numerical stability | API Doc |
| `layer_norm` | `gpt2.py:L13-17` | Layer normalization with gamma/beta | API Doc |
| `linear` | `gpt2.py:L20-21` | Linear projection (x @ w + b) | API Doc |
| `load_encoder_hparams_and_params` | `utils.py:L68-82` | Main entry point for loading model | API Doc |

---

### Alternative Implementation: gpt2_pico.py

A minimal ~40-line implementation exists in `gpt2_pico.py` (L1-62) with identical functionality but no comments. The same functions exist at these locations:

| Function | Source Location |
|----------|----------------|
| `gelu` | `gpt2_pico.py:L3-4` |
| `softmax` | `gpt2_pico.py:L6-8` |
| `layer_norm` | `gpt2_pico.py:L10-13` |
| `linear` | `gpt2_pico.py:L15-16` |
| `ffn` | `gpt2_pico.py:L18-19` |
| `attention` | `gpt2_pico.py:L21-22` |
| `mha` | `gpt2_pico.py:L24-30` |
| `transformer_block` | `gpt2_pico.py:L32-35` |
| `gpt2` | `gpt2_pico.py:L37-41` |
| `generate` | `gpt2_pico.py:L43-49` |
| `main` | `gpt2_pico.py:L51-58` |

---

**Legend:** `PENDING` = GitHub repo not yet created
