# File: `unsloth/save.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3068 |
| Functions | `print_quantization_methods`, `check_if_sentencepiece_model`, `fast_save_pickle`, `unsloth_save_model`, `install_llama_cpp_clone_non_blocking`, `install_llama_cpp_make_non_blocking`, `install_python_non_blocking`, `try_execute`, `... +25 more` |
| Imports | bitsandbytes, gc, huggingface_hub, importlib, kernels, models, ollama_template_mappers, os, pathlib, peft, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive model saving, merging, and export functionality for multiple formats (HF, GGUF, Ollama).

**Mechanism:**
- `unsloth_save_model()`: Main save function supporting three modes: "lora" (adapters only), "merged_16bit" (full model), "merged_4bit" (quantized)
- `_merge_lora()`: Dequantizes 4-bit weights and merges LoRA adapters: W + s*A*B
- Supports sharded saving with configurable max_shard_size
- GGUF export via llama.cpp integration:
  - `install_llama_cpp_*()`: Clones and builds llama.cpp with optimizations
  - `save_to_gguf()`: Converts HF model to GGUF format with various quantization methods
- `ALLOWED_QUANTS`: Dictionary of 20+ quantization methods (q4_k_m, q5_k_m, q8_0, etc.)
- `upload_to_huggingface()`: Pushes model to HF Hub with proper tags and commit messages
- Ollama integration: Creates Modelfile with correct chat templates from `OLLAMA_TEMPLATES`
- Memory-efficient saving: Uses temporary buffers, disk offloading, and garbage collection

**Significance:** The most complex module - handles the entire model export pipeline. Essential for deployment (GGUF for llama.cpp, Ollama for local inference, HF Hub for sharing).
