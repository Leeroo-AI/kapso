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

**Purpose:** Comprehensive collection of miscellaneous helper functions and wrapper classes for model preparation, quantization, and module management.

**Mechanism:** Provides utilities for k-bit training preparation, device inference, token shifting, FSDP wrapping policies, quantization config detection, tied module handling, target module matching, and wrapper classes (AuxiliaryTrainingWrapper, ModulesToSaveWrapper, TrainableTokensWrapper) for special training scenarios.

**Significance:** Essential utility toolkit that handles diverse edge cases and specialized requirements across PEFT, serving as the catch-all for helper functions that don't fit into more specific modules, particularly crucial for quantization and distributed training support.
