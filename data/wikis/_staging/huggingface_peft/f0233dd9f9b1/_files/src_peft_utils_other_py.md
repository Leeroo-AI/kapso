# File: `src/peft/utils/other.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1648 |
| Classes | `AuxiliaryTrainingWrapper`, `ModulesToSaveWrapper`, `TrainableTokensWrapper` |
| Functions | `infer_device`, `prepare_model_for_kbit_training`, `shift_tokens_right`, `fsdp_auto_wrap_policy`, `transpose`, `get_quantization_config`, `get_auto_gptq_quant_linear`, `get_gptqmodel_quant_linear`, `... +8 more` |
| Imports | __future__, accelerate, collections, constants, contextlib, copy, functools, huggingface_hub, import_utils, inspect, ... +9 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive utility module providing model manipulation functions, wrapper classes for auxiliary training (modules_to_save, trainable tokens), device inference, quantization support, and PEFT-specific helper functions.

**Mechanism:** Implements AuxiliaryTrainingWrapper base class with ModulesToSaveWrapper and TrainableTokensWrapper subclasses for training additional modules alongside adapters. Provides functions for: k-bit training preparation, module traversal (_get_submodules), adapter management (_freeze_adapter, _set_adapter), prompt learning config preparation, FSDP policy, quantization config retrieval, and attention mask creation.

**Significance:** Core infrastructure file containing essential utilities used throughout PEFT. The wrapper classes enable flexible training beyond standard adapters, while helper functions handle cross-cutting concerns like device placement, quantization compatibility (bitsandbytes, GPTQ, HQQ), and integration with accelerate/deepspeed for distributed training.
