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

**Purpose:** Registers DeepSeek model families (V3, R1, R1-Zero, R1-Distill variants) with custom naming conventions for the model registry.

**Mechanism:** Defines custom ModelInfo subclasses (`DeepseekV3ModelInfo`, `DeepseekR1ModelInfo`) that override `construct_model_name()` to implement DeepSeek-specific naming patterns (e.g., "DeepSeek-V3", "DeepSeek-R1-Zero", "DeepSeek-R1-Distill-Llama-8B"). Creates ModelMeta instances for each model family specifying org="deepseek-ai", versions, sizes, and supported quantization types (NONE, BF16, GGUF, UNSLOTH, BNB). Each registration function uses global flags to prevent duplicate registration and calls `_register_models()`. The R1-Distill variants support both Llama (8B, 70B) and Qwen (1.5B, 7B, 14B, 32B) architectures with size-specific quantization support.

**Significance:** Enables Unsloth to support the DeepSeek family of models including V3 (large-scale), R1 (reasoning), and distilled variants. DeepSeek models require special handling due to unique naming conventions and the R1 distillation strategy that produces both Llama and Qwen-based variants. The module is automatically executed on import (line 188), ensuring DeepSeek models are available in the registry.
