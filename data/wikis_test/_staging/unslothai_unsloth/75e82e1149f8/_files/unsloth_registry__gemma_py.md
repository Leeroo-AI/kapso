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

**Purpose:** Registers Google Gemma 3 model variants in both base (pretrained) and instruction-tuned versions across multiple sizes.

**Mechanism:** Defines `GemmaModelInfo` class that overrides `construct_model_name()` to format names as "gemma-3-{size}B". Creates two ModelMeta instances: `GemmaMeta3Base` for pretrained models (tagged "pt") and `GemmaMeta3Instruct` for instruction-tuned models (tagged "it"), both supporting sizes 1B, 4B, 12B, and 27B. All variants are multimodal and support quantization types NONE, BNB, and UNSLOTH, with instruction-tuned models additionally supporting GGUF format. Uses singleton pattern with `_IS_GEMMA_3_BASE_REGISTERED` and `_IS_GEMMA_3_INSTRUCT_REGISTERED` flags.

**Significance:** Essential for supporting Google's Gemma 3 model family, which includes both lightweight (1B) and large (27B) multimodal models. The separation between base and instruct variants allows proper tracking of model lineage and capabilities. Includes verification script in `__main__` block to validate all registered models against Hugging Face Hub.
