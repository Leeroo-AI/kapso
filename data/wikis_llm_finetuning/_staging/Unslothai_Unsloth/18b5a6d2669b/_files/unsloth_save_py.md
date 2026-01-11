# File: `unsloth/save.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3100 |
| Functions | `has_curl`, `print_quantization_methods`, `check_if_sentencepiece_model`, `fast_save_pickle`, `unsloth_save_model`, `install_llama_cpp_clone_non_blocking`, `install_llama_cpp_make_non_blocking`, `install_python_non_blocking`, `... +26 more` |
| Imports | bitsandbytes, gc, huggingface_hub, importlib, kernels, models, ollama_template_mappers, os, pathlib, peft, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive model saving, conversion, and export to multiple formats

**Mechanism:** Implements multiple workflows: LoRA weight merging with memory-efficient state dict management, GGUF/GGML conversion via llama.cpp integration, torchao quantization support, Ollama modelfile generation, HuggingFace Hub uploading with metadata, and support for 16bit/4bit merged models with careful RAM/VRAM management and disk caching

**Significance:** Core export utility enabling flexible model deployment across platforms (HF, Ollama, llama.cpp), multiple quantization strategies (GGUF q4_k_m/q8_0/etc, torchao), and resource-constrained environments (Kaggle/Colab) with automatic cleanup
