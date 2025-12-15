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

**Purpose:** Register DeepSeek model family variants

**Mechanism:** Defines custom ModelInfo classes (DeepseekV3ModelInfo, DeepseekR1ModelInfo) with specialized naming conventions, then creates ModelMeta configurations for each variant (V3, V3-0324, R1, R1-Zero, R1-Distill-Llama, R1-Distill-Qwen) specifying organization, sizes, quantization types, and instruct tags. Registration functions populate the global MODEL_REGISTRY.

**Significance:** Enables Unsloth to recognize and work with the complete DeepSeek model family including multiple versions (V3, R1), distilled variants, and their various quantization formats (NONE, BF16, BNB, UNSLOTH, GGUF) across different model sizes.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
