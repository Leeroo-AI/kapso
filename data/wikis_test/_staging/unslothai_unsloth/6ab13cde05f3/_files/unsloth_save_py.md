# File: `unsloth/save.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3068 |
| Functions | `print_quantization_methods`, `check_if_sentencepiece_model`, `fast_save_pickle`, `unsloth_save_model`, `install_llama_cpp_clone_non_blocking`, `install_llama_cpp_make_non_blocking`, `install_python_non_blocking`, `try_execute`, `... +25 more` |
| Imports | bitsandbytes, gc, huggingface_hub, importlib, kernels, models, ollama_template_mappers, os, pathlib, peft, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model saving and deployment preparation

**Mechanism:** Provides multiple save methods for trained models: LoRA-only (smallest), merged 16bit (for GGUF conversion), merged 4bit, and TorchAO quantization. Handles model/tokenizer serialization with HuggingFace Hub integration, GGUF conversion with llama.cpp support including quantization options, Ollama integration, and TorchAO checkpoint creation. Manages memory efficiently during merging, handles organizational repositories, and patches HuggingFace saving functions to add Unsloth metadata.

**Significance:** Enables reproducible deployment of trained models across different platforms. Supports diverse formats (native HF, GGUF for llama.cpp, Ollama) and quantization methods. The GGUF conversion pipeline is particularly important for edge deployment and CPU inference. Automatic metadata addition ensures models are tagged with Unsloth training.
