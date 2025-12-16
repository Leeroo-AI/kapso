# File: `unsloth/models/vision.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1263 |
| Classes | `FastBaseModel` |
| Functions | `unsloth_base_fast_generate` |
| Imports | _utils, contextlib, device_type, functools, gc, inspect, kernels, math, os, peft, ... +9 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides the core base class for all fast model implementations in Unsloth, with comprehensive vision-language model (VLM) support, LoRA configuration, and training/inference mode management.

**Mechanism:**
- **`FastBaseModel` class** (lines 316-1263) serves as the foundation for all model adapters:
  - **`from_pretrained`** (lines 318-907): Main factory method handling:
    - Multi-device support (CUDA/HIP/XPU detection)
    - Quantization setup (4-bit, 8-bit, 16-bit via BitsAndBytes)
    - vLLM integration for fast inference when `fast_inference=True` (lines 703-764)
    - Vision model support via `AutoModelForVision2Seq`/`AutoModelForImageTextToText`
    - Tokenizer/processor initialization with proper padding configuration
    - Full finetuning vs LoRA mode detection
    - Mixed precision configuration (FP16/BF16/FP32)
    - Embedding offloading to RAM for memory efficiency (lines 674-702)
  - **`get_peft_model`** (lines 910-1064): Configures LoRA adapters with:
    - Regex-based target module selection via `get_peft_regex`
    - Vision and language layer finetuning control
    - vLLM LoRA compatibility checks
    - QAT (Quantization-Aware Training) support
    - Gradient checkpointing setup
  - **`for_training`/`for_inference`** (lines 1162-1263): Mode switching that:
    - Toggles gradient checkpointing
    - Changes tokenizer padding side (right for training, left for inference)
    - Sets `_flag_for_generation` marker
    - Controls logits/hidden states output via environment variables
    - Manages torch compiler stance for optimization
- **`unsloth_base_fast_generate`** (lines 135-313): Enhanced generation wrapper:
  - Handles VLM pixel values and dtypes
  - Configures generation cache (static vs hybrid based on sliding window)
  - Manages mixed precision autocasting
  - Integrates with vLLM's LoRA request system
  - Cleans up Flex Attention and sliding window caches

**Significance:** This is the **foundational infrastructure** for Unsloth's entire model support system. Every model adapter (Llama, Mistral, Qwen, etc.) inherits from this class. It handles the complex orchestration of quantization, LoRA, vision capabilities, and training/inference modes. The comprehensive vLLM integration enables production-grade inference serving. The distinction between "vision.py" name and general model support reflects that this was extended from language-only to vision-language models, becoming the unified base for all modalities. Critical for understanding how Unsloth's different components integrate into a cohesive system.
