# Phase 1a: Anchoring Report

## Summary
- Workflows created: 1
- Total steps documented: 5

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| Jaymody_PicoGPT_Text_Generation | gpt2.py, utils.py, encoder.py | 5 | load_encoder_hparams_and_params, Encoder.encode, gpt2, generate, Encoder.decode |

## Coverage Summary
- Source files covered: 4/4 (100%)
- All Python files documented under single workflow

## Source Files Identified Per Workflow

### Jaymody_PicoGPT_Text_Generation
- `gpt2.py` - Main entry point with transformer implementation (gelu, softmax, layer_norm, linear, ffn, attention, mha, transformer_block, gpt2, generate, main functions)
- `utils.py` - Model download and checkpoint loading (download_gpt2_files, load_gpt2_params_from_tf_ckpt, load_encoder_hparams_and_params)
- `encoder.py` - BPE tokenizer (Encoder class with encode/decode methods, get_encoder factory)
- `gpt2_pico.py` - Minimal ~60 line implementation (same functionality as gpt2.py, alternative entry point)

## Workflow Rationale

### Why One Workflow?

PicoGPT is a minimal educational repository with a single primary use case: text generation. The Phase 0 report suggested three potential workflows:

1. **Text Generation Workflow** ✅ Created
2. **Model Loading Workflow** — This is Step 1 of Text Generation, not a standalone workflow
3. **Transformer Forward Pass** — This is Step 3 of Text Generation, an internal implementation detail

Since this repository is designed for educational purposes and has just 4 Python files (~385 lines total), having a single comprehensive workflow accurately represents the "Golden Path" users follow. Breaking into multiple workflows would fragment the natural flow of the codebase.

## Detailed API Mapping

| Step | Primary API | Supporting APIs | File |
|------|-------------|-----------------|------|
| Model_Loading | `load_encoder_hparams_and_params` | `download_gpt2_files`, `load_gpt2_params_from_tf_ckpt`, `get_encoder` | utils.py, encoder.py |
| Input_Tokenization | `Encoder.encode` | `Encoder.bpe`, `bytes_to_unicode`, `get_pairs` | encoder.py |
| Transformer_Forward_Pass | `gpt2` | `transformer_block`, `mha`, `attention`, `ffn`, `layer_norm`, `gelu`, `softmax`, `linear` | gpt2.py |
| Auto_regressive_Generation | `generate` | Uses `gpt2` internally, `np.argmax` for greedy decoding | gpt2.py |
| Output_Decoding | `Encoder.decode` | — | encoder.py |

## External Dependencies

| Dependency | Purpose | Used In |
|------------|---------|---------|
| numpy | All tensor operations | gpt2.py, utils.py |
| tensorflow | Load TF checkpoints | utils.py |
| requests | Download model files | utils.py |
| regex | Tokenizer patterns | encoder.py |
| tqdm | Progress bars | gpt2.py, utils.py |
| fire | CLI interface | gpt2.py |

## Notes for Phase 1b (Enrichment)

### Files Requiring Line-by-Line Tracing
1. **gpt2.py:L73-83** - The `gpt2()` function implements the complete forward pass
2. **gpt2.py:L38-60** - Multi-head attention with causal masking
3. **encoder.py:L60-99** - BPE algorithm implementation
4. **utils.py:L44-65** - TensorFlow checkpoint parsing and weight restructuring

### External APIs to Document
- `tf.train.list_variables` / `tf.train.load_variable` — TensorFlow checkpoint reading
- `np.argmax` — Greedy decoding strategy
- `fire.Fire` — CLI generation pattern

### Key Implementation Patterns
1. **Functional transformer** — All operations are pure functions with explicit tensor shapes in comments
2. **Nested dict weights** — `params["blocks"][i]["attn"]["c_attn"]["w"]` structure
3. **BPE with byte fallback** — Avoids UNK tokens via byte-to-unicode mapping
4. **Lazy downloading** — Model files downloaded only when checkpoint not found

### Educational Design Notes
This repository prioritizes readability over efficiency:
- Pure NumPy (no framework abstractions)
- Explicit tensor shape annotations
- Two versions: verbose (`gpt2.py`) and minimal (`gpt2_pico.py`)
- No batching, sampling, or optimization — just the core algorithm
