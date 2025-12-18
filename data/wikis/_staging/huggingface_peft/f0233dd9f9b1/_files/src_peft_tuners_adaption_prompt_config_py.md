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

**Purpose:** Adaption prompt configuration

**Mechanism:** Defines AdaptionPromptConfig with target_modules (attention submodules), adapter_len (number of tokens), adapter_layers (number from top). TRANSFORMERS_MODEL_CONFIG maps model types (llama, mistral, gpt2) to compute_query_states functions and projection layer names. prepare_config() validates model type compatibility.

**Significance:** Core configuration for adaption prompt tuning. ModelTypeConfig namedtuple enables multi-architecture support (LLaMA, Mistral, GPT-2) with architecture-specific attention handling.
