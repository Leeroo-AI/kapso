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

**Purpose:** Registers DeepSeek model variants including V3, R1, R1-Zero, and R1-Distill versions (both Llama and Qwen based) with their respective quantization types and configurations.

**Mechanism:** Defines custom ModelInfo subclasses (`DeepseekV3ModelInfo`, `DeepseekR1ModelInfo`) that override `construct_model_name()` to follow DeepSeek's naming conventions. Creates ModelMeta instances for each DeepSeek variant (V3, V3-0324, R1, R1-Zero, R1-Distill-Llama, R1-Distill-Qwen) specifying organizations, versions, sizes, quantization types (including NONE, BF16, BNB, UNSLOTH, GGUF), and whether models are multimodal. Each registration function uses singleton pattern with global flags to prevent duplicate registration.

**Significance:** Critical for supporting DeepSeek's extensive model family, particularly the recent R1 reasoning models and their distilled variants. Handles size-specific quantization options (e.g., different quant types for 8B vs 70B models) and includes utility function `_list_deepseek_r1_distill_models()` for discovering models on Hugging Face Hub. The module runs registration on import and includes verification logic in `__main__`.
