# File: `unsloth/registry/_deepseek.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 206 |
| Classes | `DeepseekV3ModelInfo`, `DeepseekR1ModelInfo` |
| Functions | `register_deepseek_v3_models`, `register_deepseek_v3_0324_models`, `register_deepseek_r1_models`, `register_deepseek_r1_zero_models`, `register_deepseek_r1_distill_llama_models`, `register_deepseek_r1_distill_qwen_models`, `register_deepseek_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines model metadata and registration logic for DeepSeek model variants including V3, R1, R1-Zero, and R1-Distill models.

**Mechanism:** Provides two custom `ModelInfo` subclasses (`DeepseekV3ModelInfo`, `DeepseekR1ModelInfo`) with specialized `construct_model_name()` methods for DeepSeek's naming conventions. Defines `ModelMeta` configurations for each variant specifying organization, sizes, quantization types, and multimodality. Uses per-model-family registration flags to prevent duplicate registrations. R1-Distill variants support both Llama and Qwen distillations with size-specific quantization options (e.g., 1.5B-32B for Qwen, 8B-70B for Llama).

**Significance:** Enables Unsloth to support the DeepSeek model family with proper HuggingFace Hub path construction. The file auto-registers models at import time and includes a utility function `_list_deepseek_r1_distill_models()` to discover distilled variants from the Hub.
