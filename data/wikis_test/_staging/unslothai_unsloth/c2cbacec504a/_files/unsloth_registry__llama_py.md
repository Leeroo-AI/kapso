# File: `unsloth/registry/_llama.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 125 |
| Classes | `LlamaModelInfo`, `LlamaVisionModelInfo` |
| Functions | `register_llama_3_1_models`, `register_llama_3_2_models`, `register_llama_3_2_vision_models`, `register_llama_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers Meta's Llama 3.1 and 3.2 models including base, instruction-tuned, and vision variants with size-specific quantization.

**Mechanism:** Defines three ModelInfo subclasses (LlamaModelInfo, LlamaVisionModelInfo) for naming conventions; creates separate ModelMeta for Llama 3.1 (8B), Llama 3.2 base/instruct (1B, 3B), and vision (11B, 90B with size-dependent quantization constraints).

**Significance:** Provides comprehensive support for Meta's flagship Llama models with careful quantization constraints for large vision variants.
