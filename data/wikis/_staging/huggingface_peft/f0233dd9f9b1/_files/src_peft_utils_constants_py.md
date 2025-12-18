# File: `src/peft/utils/constants.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 362 |
| Functions | `bloom_model_postprocess_past_key_value`, `starcoder_model_postprocess_past_key_value` |
| Imports | packaging, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines default target modules mappings for various PEFT methods across different transformer architectures, plus postprocessing functions for specific models and file naming constants.

**Mechanism:** Provides extensive dictionaries mapping model types (e.g., "llama", "gpt2", "bert") to their default target modules for each PEFT method (LoRA, AdaLoRA, IA3, BOFT, etc.). Includes helper functions for model-specific postprocessing (Bloom, StarCoder) and constants for file names (WEIGHTS_NAME, CONFIG_NAME, etc.).

**Significance:** Critical configuration component that enables automatic target module selection for PEFT methods based on model architecture. Eliminates the need for users to manually specify which modules to adapt, supporting 30+ model architectures and 20+ PEFT methods with intelligent defaults.
