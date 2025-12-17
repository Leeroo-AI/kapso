# File: `unsloth/save.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3086 |
| Functions | `print_quantization_methods`, `check_if_sentencepiece_model`, `fast_save_pickle`, `unsloth_save_model`, `install_llama_cpp_clone_non_blocking`, `install_llama_cpp_make_non_blocking`, `install_python_non_blocking`, `try_execute`, `... +25 more` |
| Imports | bitsandbytes, gc, huggingface_hub, importlib, kernels, models, ollama_template_mappers, os, pathlib, peft, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Handles model saving in multiple formats (HuggingFace, GGUF, LoRA, merged) with quantization support and integration with Ollama and HuggingFace Hub.

**Mechanism:** Implements unsloth_save_model() as main entry point dispatching to format-specific functions. Provides save_to_gguf() for llama.cpp format with optional quantization, unsloth_save_pretrained_merged() for merged weights, and LoRA saving utilities. Manages llama.cpp installation, GGUF conversion/quantization, and HuggingFace Hub uploads. Generates Ollama modelfiles for local model serving.

**Significance:** Unifies model saving across different formats that users need (GGUF for local inference, merged weights for deployment, LoRA for sharing fine-tuning adapters). Automates complex processes like model quantization and repository creation.
