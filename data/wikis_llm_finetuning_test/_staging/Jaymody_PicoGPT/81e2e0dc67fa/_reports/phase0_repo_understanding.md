# Phase 0: Repository Understanding Report

## Summary
- Files explored: 4/4
- Completion: 100%

## Key Discoveries

### Main Entry Points
- **`gpt2.py`** - Primary entry point for users. Runs GPT-2 inference with a well-commented, educational implementation (~120 lines). Executable via `python gpt2.py --prompt "Your text here"`.
- **`gpt2_pico.py`** - Alternative "code golf" entry point with identical functionality in ~60 lines.

### Core Modules
1. **`encoder.py`** - BPE tokenizer (copied from OpenAI's GPT-2 repo). Handles text↔token conversion.
2. **`utils.py`** - Model loading infrastructure. Downloads GPT-2 weights from OpenAI's Azure blob storage and parses TensorFlow checkpoints into NumPy arrays.

### Architecture Patterns
- **Pure NumPy implementation** - No PyTorch/TensorFlow for inference, only for loading weights
- **Functional style** - All transformer operations are pure functions (gelu, softmax, layer_norm, linear, ffn, attention, mha, transformer_block)
- **Weight dict structure** - Parameters organized as nested dicts: `params["blocks"][i]["attn"]["c_attn"]["w"]`
- **CLI via python-fire** - Simple `fire.Fire(main)` pattern for command-line interface
- **Greedy decoding** - Uses `np.argmax(logits[-1])` for token selection (no sampling)

### File Dependencies
```
gpt2.py / gpt2_pico.py
    └── utils.py (load_encoder_hparams_and_params)
            └── encoder.py (get_encoder)
```

## Recommendations for Next Phase

### Suggested Workflows to Document
1. **Text Generation Workflow** - `prompt → encode → forward pass → decode → output`
2. **Model Loading Workflow** - `download files → load checkpoint → restructure params → initialize encoder`
3. **Transformer Forward Pass** - `embeddings → N blocks(attention + ffn) → layer_norm → logits`

### Key APIs to Trace
- `gpt2(inputs, wte, wpe, blocks, ln_f, n_head)` - Main forward pass
- `generate(inputs, params, n_head, n_tokens_to_generate)` - Auto-regressive generation loop
- `Encoder.encode(text)` / `Encoder.decode(tokens)` - Tokenization interface
- `load_encoder_hparams_and_params(model_size, models_dir)` - Model initialization

### Important Files for Anchoring Phase
1. **`gpt2.py`** - Central anchor with all transformer concepts implemented and commented
2. **`encoder.py`** - BPE tokenization anchor
3. **`utils.py`** - Weight loading anchor (TF checkpoint → NumPy dict)

### Educational Value
This repository is notable for its educational design:
- Two versions allow comparison (verbose vs minimal)
- Pure NumPy makes the math explicit
- No framework abstraction hides the transformer operations
- Each function has clear tensor shape annotations in comments
