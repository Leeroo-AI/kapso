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

**Purpose:** Registers Deepseek model family variants including V3, R1, and R1-Distill models with their specific naming conventions and quantization options.

**Mechanism:** Defines custom ModelInfo subclasses (DeepseekV3ModelInfo, DeepseekR1ModelInfo) that override construct_model_name() to implement Deepseek-specific naming patterns. Creates ModelMeta instances for six model lines: DeepSeek-V3 (bf16/none quants), DeepSeek-V3-0324 (gguf/none), DeepSeek-R1 (bf16/gguf/none), DeepSeek-R1-Zero (gguf/none), DeepSeek-R1-Distill-Llama (8B/70B with size-specific quants), and DeepSeek-R1-Distill-Qwen (1.5B/7B/14B/32B with size-specific quants including unsloth/bnb/gguf). Each model line has a dedicated registration function with a global flag to prevent duplicate registration, coordinated by register_deepseek_models().

**Significance:** Supports the most complex model family in the registry with multiple versions, distillation variants, and size-dependent quantization strategies, reflecting Deepseek's diverse model offerings including reasoning models (R1) and their distilled versions across different base architectures.
