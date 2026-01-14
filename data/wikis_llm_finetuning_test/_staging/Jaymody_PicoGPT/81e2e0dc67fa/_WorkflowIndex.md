# Workflow Index: Jaymody_PicoGPT

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Repository Building).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Rough APIs | GitHub URL |
|----------|-------|------------|------------|
| Text_Generation | 5 | load_encoder_hparams_and_params, Encoder.encode, gpt2, generate, Encoder.decode | PENDING |

---

## Workflow: Jaymody_PicoGPT_Text_Generation

**File:** [→](./workflows/Jaymody_PicoGPT_Text_Generation.md)
**Description:** End-to-end GPT-2 text generation using pure NumPy implementation.
**GitHub URL:** PENDING

### Steps Overview

| # | Step Name | Rough API | Related Files |
|---|-----------|-----------|---------------|
| 1 | Model_Loading | `load_encoder_hparams_and_params` | utils.py, encoder.py |
| 2 | Input_Tokenization | `Encoder.encode` | encoder.py |
| 3 | Transformer_Forward_Pass | `gpt2` | gpt2.py |
| 4 | Auto_regressive_Generation | `generate` | gpt2.py |
| 5 | Output_Decoding | `Encoder.decode` | encoder.py |

### Source Files (for enrichment)

- `gpt2.py` - Main inference script with transformer implementation and generation loop
- `utils.py` - Model download and TensorFlow checkpoint loading utilities
- `encoder.py` - BPE tokenizer implementation (Encoder class)
- `gpt2_pico.py` - Minimal version of gpt2.py (~60 lines)

---

### Step 1: Model_Loading

| Attribute | Value |
|-----------|-------|
| **API Call** | `load_encoder_hparams_and_params(model_size: str, models_dir: str) -> Tuple[Encoder, dict, dict]` |
| **Source Location** | `utils.py:L68-82` |
| **External Dependencies** | `tensorflow`, `requests`, `numpy`, `json`, `tqdm`, `encoder` |
| **Key Parameters** | `model_size: str` - GPT-2 model variant ("124M", "355M", "774M", "1558M"), `models_dir: str` - local directory for model files |
| **Inputs** | Model size specification and local storage directory path |
| **Outputs** | Tuple of (Encoder instance, hparams dict, params dict) for tokenization and inference |

**Supporting Functions:**

| Function | Signature | Source | Purpose |
|----------|-----------|--------|---------|
| `download_gpt2_files` | `download_gpt2_files(model_size: str, model_dir: str) -> None` | `utils.py:L13-41` | Downloads GPT-2 checkpoint/vocab files from OpenAI Azure storage |
| `load_gpt2_params_from_tf_ckpt` | `load_gpt2_params_from_tf_ckpt(tf_ckpt_path: str, hparams: dict) -> dict` | `utils.py:L44-65` | Parses TensorFlow checkpoint into nested dict of NumPy arrays |
| `get_encoder` | `get_encoder(model_name: str, models_dir: str) -> Encoder` | `encoder.py:L114-120` | Loads encoder.json and vocab.bpe to create Encoder instance |

---

### Step 2: Input_Tokenization

| Attribute | Value |
|-----------|-------|
| **API Call** | `Encoder.encode(text: str) -> List[int]` |
| **Source Location** | `encoder.py:L101-106` |
| **External Dependencies** | `regex` |
| **Key Parameters** | `text: str` - input text to tokenize |
| **Inputs** | Raw text string (user prompt) |
| **Outputs** | List of integer token IDs |

**Supporting Functions:**

| Function | Signature | Source | Purpose |
|----------|-----------|--------|---------|
| `Encoder.__init__` | `Encoder.__init__(encoder: dict, bpe_merges: list, errors: str = "replace")` | `encoder.py:L48-58` | Initializes encoder/decoder dicts and BPE ranking |
| `Encoder.bpe` | `Encoder.bpe(token: str) -> str` | `encoder.py:L60-99` | Applies BPE merging to a single token |
| `bytes_to_unicode` | `bytes_to_unicode() -> dict` | `encoder.py:L12-32` | Creates UTF-8 byte to unicode mapping |
| `get_pairs` | `get_pairs(word: tuple) -> set` | `encoder.py:L35-44` | Returns symbol pairs for BPE merging |

---

### Step 3: Transformer_Forward_Pass

| Attribute | Value |
|-----------|-------|
| **API Call** | `gpt2(inputs: List[int], wte: np.ndarray, wpe: np.ndarray, blocks: List[dict], ln_f: dict, n_head: int) -> np.ndarray` |
| **Source Location** | `gpt2.py:L73-83` |
| **External Dependencies** | `numpy` |
| **Key Parameters** | `inputs: List[int]` - token IDs, `wte: np.ndarray` - token embeddings, `wpe: np.ndarray` - positional embeddings, `blocks: List[dict]` - transformer layer params, `ln_f: dict` - final layer norm params, `n_head: int` - number of attention heads |
| **Inputs** | Token IDs and pre-trained model parameters |
| **Outputs** | Logits array of shape [n_seq, n_vocab] |

**Supporting Functions:**

| Function | Signature | Source | Purpose |
|----------|-----------|--------|---------|
| `gelu` | `gelu(x: np.ndarray) -> np.ndarray` | `gpt2.py:L4-5` | GELU activation function |
| `softmax` | `softmax(x: np.ndarray) -> np.ndarray` | `gpt2.py:L8-10` | Numerically stable softmax |
| `layer_norm` | `layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray` | `gpt2.py:L13-17` | Layer normalization with gamma/beta |
| `linear` | `linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray` | `gpt2.py:L20-21` | Linear projection (matmul + bias) |
| `ffn` | `ffn(x: np.ndarray, c_fc: dict, c_proj: dict) -> np.ndarray` | `gpt2.py:L24-31` | Feed-forward network (up-project → GELU → down-project) |
| `attention` | `attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray` | `gpt2.py:L34-35` | Scaled dot-product attention with causal mask |
| `mha` | `mha(x: np.ndarray, c_attn: dict, c_proj: dict, n_head: int) -> np.ndarray` | `gpt2.py:L38-60` | Multi-head self-attention |
| `transformer_block` | `transformer_block(x: np.ndarray, mlp: dict, attn: dict, ln_1: dict, ln_2: dict, n_head: int) -> np.ndarray` | `gpt2.py:L63-70` | Single transformer block (attention + FFN with residuals) |

---

### Step 4: Auto_regressive_Generation

| Attribute | Value |
|-----------|-------|
| **API Call** | `generate(inputs: List[int], params: dict, n_head: int, n_tokens_to_generate: int) -> List[int]` |
| **Source Location** | `gpt2.py:L86-94` |
| **External Dependencies** | `numpy`, `tqdm` |
| **Key Parameters** | `inputs: List[int]` - initial token IDs, `params: dict` - model parameters, `n_head: int` - attention heads, `n_tokens_to_generate: int` - output length |
| **Inputs** | Token IDs from encoding step, model params from loading step |
| **Outputs** | List of generated token IDs (length = n_tokens_to_generate) |

**Generation Loop Details:**
- Uses greedy sampling (`np.argmax` on final logit position)
- Appends each generated token to input for next iteration
- Shows progress bar via `tqdm`

---

### Step 5: Output_Decoding

| Attribute | Value |
|-----------|-------|
| **API Call** | `Encoder.decode(tokens: List[int]) -> str` |
| **Source Location** | `encoder.py:L108-111` |
| **External Dependencies** | — (pure Python) |
| **Key Parameters** | `tokens: List[int]` - token IDs to decode |
| **Inputs** | List of token IDs from generation step |
| **Outputs** | Decoded text string |

**Decoding Process:**
1. Maps token IDs to BPE tokens via `self.decoder`
2. Joins tokens into single string
3. Converts unicode back to UTF-8 bytes via `self.byte_decoder`
4. Decodes bytes to string with error handling

---

### Implementation Extraction Guide

| Step | API | Source | Dependencies | Type |
|------|-----|--------|--------------|------|
| Model_Loading | `load_encoder_hparams_and_params` | `utils.py:L68-82` | tensorflow, requests, numpy, tqdm | API Doc |
| Model_Loading | `download_gpt2_files` | `utils.py:L13-41` | requests, tqdm | API Doc |
| Model_Loading | `load_gpt2_params_from_tf_ckpt` | `utils.py:L44-65` | tensorflow, numpy | API Doc |
| Model_Loading | `get_encoder` | `encoder.py:L114-120` | json | API Doc |
| Input_Tokenization | `Encoder.encode` | `encoder.py:L101-106` | regex | API Doc |
| Input_Tokenization | `Encoder.bpe` | `encoder.py:L60-99` | — | API Doc |
| Input_Tokenization | `bytes_to_unicode` | `encoder.py:L12-32` | — | API Doc |
| Input_Tokenization | `get_pairs` | `encoder.py:L35-44` | — | API Doc |
| Transformer_Forward_Pass | `gpt2` | `gpt2.py:L73-83` | numpy | API Doc |
| Transformer_Forward_Pass | `transformer_block` | `gpt2.py:L63-70` | numpy | API Doc |
| Transformer_Forward_Pass | `mha` | `gpt2.py:L38-60` | numpy | API Doc |
| Transformer_Forward_Pass | `attention` | `gpt2.py:L34-35` | numpy | API Doc |
| Transformer_Forward_Pass | `ffn` | `gpt2.py:L24-31` | numpy | API Doc |
| Transformer_Forward_Pass | `layer_norm` | `gpt2.py:L13-17` | numpy | API Doc |
| Transformer_Forward_Pass | `gelu` | `gpt2.py:L4-5` | numpy | API Doc |
| Transformer_Forward_Pass | `softmax` | `gpt2.py:L8-10` | numpy | API Doc |
| Transformer_Forward_Pass | `linear` | `gpt2.py:L20-21` | numpy | API Doc |
| Auto_regressive_Generation | `generate` | `gpt2.py:L86-94` | tqdm, numpy | API Doc |
| Output_Decoding | `Encoder.decode` | `encoder.py:L108-111` | — | API Doc |

---

**Legend:** `PENDING` = GitHub repo not yet created

**Implementation Types:**
- **API Doc:** Function/class in this repo
- **Wrapper Doc:** External library with repo-specific usage
- **Pattern Doc:** User-defined interface/pattern
- **External Tool Doc:** CLI or external tool
