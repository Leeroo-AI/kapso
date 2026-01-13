# File: `unsloth/save.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3100 |
| Functions | `has_curl`, `print_quantization_methods`, `check_if_sentencepiece_model`, `fast_save_pickle`, `unsloth_save_model`, `install_llama_cpp_clone_non_blocking`, `install_llama_cpp_make_non_blocking`, `install_python_non_blocking`, `... +26 more` |
| Imports | bitsandbytes, gc, huggingface_hub, importlib, kernels, models, ollama_template_mappers, os, pathlib, peft, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive model saving and export functionality supporting multiple formats (HuggingFace, GGUF, merged 16-bit/4-bit) and destinations (local, HuggingFace Hub, Ollama).

**Mechanism:** The main `unsloth_save_model()` function coordinates saving with options for LoRA adapter merging, quantization, and format conversion. Integrates with llama.cpp for GGUF quantization (supports 30+ quantization methods). Handles SentencePiece tokenizer serialization, generates Ollama Modelfiles, and manages HuggingFace Hub uploads. Uses non-blocking installation helpers for llama.cpp compilation.

**Significance:** Core component - the primary interface for exporting fine-tuned models. Critical for deployment workflows, enabling users to save models in production-ready formats like GGUF for local inference with Ollama/llama.cpp.
