# File: `src/peft/tuners/adaption_prompt/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 88 |
| Classes | `AdaptionPromptConfig` |
| Functions | `prepare_config` |
| Imports | collections, dataclasses, peft, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration for Adaption Prompt with model-specific mappings

**Mechanism:** AdaptionPromptConfig stores adapter_len (number of prompt tokens), adapter_layers (how many top layers), target_modules (attention submodules). TRANSFORMERS_MODEL_CONFIG maps model types (llama/mistral/gpt2) to their compute_query_states functions and projection layer names. prepare_config() auto-fills target_modules based on model type

**Significance:** Provides model-agnostic configuration interface for adaption prompts while handling architectural differences between transformer variants (LLaMA uses separate k/v projections, GPT-2 uses combined c_attn)
