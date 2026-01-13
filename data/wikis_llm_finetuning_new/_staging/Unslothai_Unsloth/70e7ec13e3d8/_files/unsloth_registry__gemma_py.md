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

**Purpose:** Defines model metadata and registration logic for Google's Gemma 3 model family (base and instruction-tuned variants).

**Mechanism:** Provides `GemmaModelInfo` class that constructs model names in the format `gemma-{version}-{size}B`. Defines two `ModelMeta` configurations: `GemmaMeta3Base` for pretrained models (tag "pt") and `GemmaMeta3Instruct` for instruction-tuned models (tag "it"). Both support sizes 1B, 4B, 12B, and 27B. All Gemma models are marked as multimodal. Instruct variants additionally support GGUF quantization.

**Significance:** Enables Unsloth support for Gemma 3 models with proper naming conventions. The "pt"/"it" tags follow Google's convention for pretrained vs instruction-tuned variants. Includes standalone verification via `__main__` to validate registered models exist on HuggingFace Hub.
