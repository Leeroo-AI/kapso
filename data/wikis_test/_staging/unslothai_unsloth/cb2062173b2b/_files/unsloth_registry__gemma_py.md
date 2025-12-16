# File: `unsloth/registry/_gemma.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 74 |
| Classes | `GemmaModelInfo` |
| Functions | `register_gemma_3_base_models`, `register_gemma_3_instruct_models`, `register_gemma_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers Google's Gemma 3 model family (base and instruction-tuned variants) with standard naming conventions.

**Mechanism:** Defines `GemmaModelInfo` class that constructs model names in format "gemma-3-{size}B" (e.g., "gemma-3-1B", "gemma-3-27B"). Creates two ModelMeta instances: `GemmaMeta3Base` for base models with "pt" (pre-trained) tag supporting sizes 1B/4B/12B/27B with NONE/BNB/UNSLOTH quantization, and `GemmaMeta3Instruct` for instruction-tuned models with "it" tag supporting same sizes plus GGUF quantization. Both are marked as multimodal (is_multimodal=True). Registration functions use global flags to ensure one-time registration per variant.

**Significance:** Provides access to Google's Gemma 3 series, which are efficient open models with multimodal capabilities. The distinction between base and instruct variants allows users to choose between foundation models for further training vs. ready-to-use instruction-following models. The multimodal flag indicates these models can handle both text and visual inputs.
