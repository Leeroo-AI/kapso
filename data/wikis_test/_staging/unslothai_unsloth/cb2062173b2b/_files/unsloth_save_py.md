# File: `unsloth/save.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3068 |
| Functions | `print_quantization_methods`, `check_if_sentencepiece_model`, `fast_save_pickle`, `unsloth_save_model`, `install_llama_cpp_clone_non_blocking`, `install_llama_cpp_make_non_blocking`, `install_python_non_blocking`, `try_execute`, `... +25 more` |
| Imports | bitsandbytes, gc, huggingface_hub, importlib, kernels, models, ollama_template_mappers, os, pathlib, peft, ... +12 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Handles model saving and export in multiple formats (LoRA, merged 16-bit, merged 4-bit, GGUF) with HuggingFace Hub integration. Manages LoRA weight merging, quantization, and llama.cpp compilation for GGUF conversion.

**Mechanism:**
- **Save methods**:
  - `lora`: Saves only LoRA adapter weights (smallest, requires base model)
  - `merged_16bit`: Merges LoRA into base model in FP16/BF16 (compatible with most tools)
  - `merged_4bit`: Merges LoRA and requantizes to 4-bit (space-efficient, slight accuracy loss)
- **LoRA merging** (`_merge_lora`):
  - Dequantizes 4-bit base weights using `fast_dequantize()`
  - Computes merged weights: `W_merged = W_base + s * A^T * B^T` where s=lora_alpha/r
  - Validates finite values to detect merge failures
- **GGUF conversion**:
  - Installs/compiles llama.cpp on-demand with progress tracking
  - Calls `convert_to_gguf()` to create initial F16/F32 GGUF
  - Uses `quantize_gguf()` for target quantization (q4_k_m, q8_0, etc.)
  - Supports 20+ quantization methods from llama.cpp
- **Memory management**:
  - Monitors system memory via `psutil` to prevent OOM
  - Deletes cached HuggingFace repos to free space (useful on Kaggle/Colab)
  - Shards large models to stay under max_shard_size (default 5GB)
- **SentencePiece handling**: Fixes tokenizer.model when extending vocabulary with new tokens
- **HuggingFace Hub**:
  - `create_huggingface_repo()`: Creates repo with correct tags and README
  - Uploads GGUF files with proper quantization naming (e.g., `model-Q4_K_M.gguf`)
  - Supports private repos, custom commits, and PR creation
- **Ollama integration**: Generates Modelfile with appropriate templates from `ollama_template_mappers`
- **TorchAO support**: Handles conversion of TorchAO quantized models via `_convert_torchao_model()`

**Significance:** This module enables the complete training-to-deployment pipeline. The multi-format saving is critical for flexibility: LoRA for iteration, merged_16bit for production, GGUF for local deployment (llama.cpp/Ollama/GPT4All). The on-demand llama.cpp compilation makes GGUF conversion accessible without manual setup. Memory management prevents OOM on resource-constrained environments (Kaggle, Colab). Hub integration automates model sharing. This file embodies Unsloth's philosophy of making model deployment as easy as training.
