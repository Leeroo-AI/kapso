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

**Purpose:** Registers Google Gemma 3 model variants in base and instruction-tuned configurations with multimodal support.

**Mechanism:** Defines GemmaModelInfo class for consistent naming (e.g., "gemma-3-1B"), creates separate ModelMeta for base (pt) and instruction (it) versions, supports four size options (1B, 4B, 12B, 27B) with multimodal flag and quantization options.

**Significance:** Enables integration of Google's latest multimodal Gemma models with flexible quantization support for different deployment scenarios.
