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

**Purpose:** Registers Deepseek model variants including V3, V3-0324, R1, R1-Zero, and distilled versions (Llama and Qwen based).

**Mechanism:** Defines custom ModelInfo classes for version-specific naming conventions, creates ModelMeta configurations for each variant with specific quantization support per size, and provides registration functions with size-dependent quantization type mappings.

**Significance:** Extends model registry with cutting-edge Deepseek reasoning models and distilled variants, enabling efficient deployment of advanced LLMs across varying resource constraints.
