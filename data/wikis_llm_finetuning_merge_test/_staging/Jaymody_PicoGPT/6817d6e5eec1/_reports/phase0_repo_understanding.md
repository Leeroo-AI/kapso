# Phase 0: Repository Understanding Report

## Summary
- Files explored: 4/4
- Completion: 100%

## Key Discoveries

### Main Entry Points
- **`gpt2.py`**: Primary entry point with CLI interface via python-fire. Contains the fully documented GPT-2 implementation intended for educational use.
- **`gpt2_pico.py`**: Alternative minimal entry point - same functionality compressed to ~40 lines, demonstrating the "pico" concept.

### Core Modules Identified
1. **`gpt2.py`** - Complete GPT-2 transformer architecture in pure NumPy with detailed comments
2. **`gpt2_pico.py`** - Minimal version of the same implementation (~40 lines of core code)
3. **`encoder.py`** - BPE tokenizer copied from OpenAI's GPT-2 repository
4. **`utils.py`** - Model loading utilities (download from Azure, convert TensorFlow checkpoints to NumPy)

### Architecture Patterns Observed
- **Pure NumPy Implementation**: No PyTorch/TensorFlow for inference - all computations done with NumPy
- **Pre-norm Transformer**: Layer normalization applied before attention and FFN (not after)
- **Causal Masking**: Uses lower-triangular mask with -1e10 for future positions
- **Greedy Decoding**: Simple argmax sampling for text generation
- **TensorFlow Checkpoint Loading**: Converts official OpenAI TF checkpoints to NumPy arrays
- **Two Implementation Styles**: Documented version (gpt2.py) vs minimal version (gpt2_pico.py)

### Dependencies
- `numpy` - Core computation
- `tensorflow` - Only for loading official checkpoints (not for inference)
- `requests` - Downloading model files
- `regex` - BPE tokenization patterns
- `fire` - CLI interface
- `tqdm` - Progress bars

## File Relationships

```
main() in gpt2.py/gpt2_pico.py
    ├── utils.load_encoder_hparams_and_params()
    │   ├── utils.download_gpt2_files() [if needed]
    │   ├── encoder.get_encoder()
    │   └── utils.load_gpt2_params_from_tf_ckpt()
    ├── encoder.encode() [tokenize prompt]
    ├── generate()
    │   └── gpt2() [forward pass through transformer]
    └── encoder.decode() [detokenize output]
```

## Recommendations for Next Phase

### Suggested Workflows to Document
1. **Text Generation Workflow**: End-to-end from prompt to generated text
2. **Model Loading Workflow**: Downloading and converting official GPT-2 weights
3. **Tokenization Workflow**: BPE encoding/decoding process

### Key APIs to Trace
- `main(prompt, n_tokens_to_generate, model_size, models_dir)` - Entry point
- `gpt2(inputs, wte, wpe, blocks, ln_f, n_head)` - Forward pass
- `generate(inputs, params, n_head, n_tokens_to_generate)` - Autoregressive generation
- `Encoder.encode(text)` / `Encoder.decode(tokens)` - Tokenization

### Important Files for Anchoring Phase
- **`gpt2.py`**: Main implementation to anchor, contains detailed shape comments
- **`encoder.py`**: Core tokenization logic
- **`utils.py`**: Critical for understanding how official weights are loaded

### Conceptual Topics to Cover
1. Transformer architecture (attention, FFN, residuals, layer norm)
2. Byte-pair encoding (BPE) tokenization
3. Autoregressive text generation
4. Pre-norm vs post-norm transformers
5. Model weight conversion (TF → NumPy)
