# File: `unsloth/models/_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 2453 |
| Classes | `HideLoggingMessage`, `_RaiseUninitialized`, `RaiseUninitialized`, `EmptyLogits`, `TorchAOConfig` |
| Functions | `extract_quant_model_param_count`, `get_model_param_count`, `patch_mistral_nemo_config`, `is_big_gpu`, `torch_compile_kwargs`, `patch_regional_compilation`, `prepare_model_for_kbit_training`, `has_internet`, `... +25 more` |
| Imports | accelerate, bitsandbytes, contextlib, dataclasses, device_type, functools, importlib, inspect, logging, math, ... +16 more |

## Understanding

**Status:** Explored

**Purpose:** Central utilities module providing version constants, hardware detection, attention backend setup, gradient checkpointing patches, RoPE scaling implementations, and model preparation functions used throughout Unsloth.

**Mechanism:** Defines `__version__ = "2026.1.2"` and detects hardware/software capabilities including bfloat16 support (`is_bfloat16_supported`), Flash Attention availability (`HAS_FLASH_ATTENTION`, `HAS_FLASH_ATTENTION_SOFTCAPPING`), and xformers integration. Implements smart gradient checkpointing (`patch_unsloth_smart_gradient_checkpointing`), RoPE scaling patches (`patch_llama_rope_scaling`, `patch_linear_scaling`), and model training preparation (`prepare_model_for_kbit_training`). Provides `TorchAOConfig` dataclass for Quantization-Aware Training configuration. Includes logging suppression utilities (`HideLoggingMessage`) and torch.compile configuration options.

**Significance:** Core utility component - nearly every other module in Unsloth imports from this file. It provides the foundational infrastructure for hardware detection, attention backend selection, and model optimization patches that enable Unsloth's performance improvements.
