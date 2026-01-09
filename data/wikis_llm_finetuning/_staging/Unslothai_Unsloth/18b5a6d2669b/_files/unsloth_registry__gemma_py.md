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

**Purpose:** Registers Google Gemma 3 model variants with base and instruction-tuned configurations.

**Mechanism:** Defines GemmaModelInfo class that constructs names in pattern "gemma-{version}-{size}B" plus instruct/quant tags. Creates two ModelMeta instances: GemmaMeta3Base for pretrained models (pt tag, 1B/4B/12B/27B sizes, none/bnb/unsloth quants) and GemmaMeta3Instruct for instruction-tuned models (it tag, same sizes, adds gguf quant option). Both are marked as multimodal (is_multimodal=True). Provides register_gemma_3_base_models() and register_gemma_3_instruct_models() functions with global flags, coordinated by register_gemma_models().

**Significance:** Handles Google's Gemma 3 family which includes multimodal capabilities, distinguishing between base pretrained (pt) and instruction-tuned (it) variants across multiple model sizes with comprehensive quantization support.
