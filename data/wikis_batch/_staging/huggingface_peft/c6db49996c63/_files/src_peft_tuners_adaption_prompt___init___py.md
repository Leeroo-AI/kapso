# File: `src/peft/tuners/adaption_prompt/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization for Adaption Prompt tuner module

**Mechanism:** Exports core components (AdaptedAttention, AdaptionPromptConfig, AdaptionPromptModel) and registers the PEFT method with name "adaption_prompt"

**Significance:** Entry point for LLaMA-Adapter style prompt tuning - enables efficient fine-tuning by injecting learnable prompt tokens into attention layers of frozen LLMs
