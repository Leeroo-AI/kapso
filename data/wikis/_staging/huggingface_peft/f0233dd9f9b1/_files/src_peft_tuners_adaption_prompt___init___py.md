# File: `src/peft/tuners/adaption_prompt/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Adaption prompt module initialization

**Mechanism:** Exports AdaptedAttention, AdaptionPromptConfig, AdaptionPromptModel classes and registers "adaption_prompt" as PEFT method for LLaMA-Adapter style tuning.

**Significance:** Entry point for adaption prompt tuning method based on LLaMA-Adapter (https://huggingface.co/papers/2303.16199). Enables prompt-based adaptation via learnable tokens in attention layers.
