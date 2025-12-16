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

**Purpose:** Registers Meta's Llama 3.1 and 3.2 model families including text-only and vision variants with appropriate naming conventions.

**Mechanism:** Defines two ModelInfo classes: `LlamaModelInfo` for standard text models using format "Llama-{version}-{size}B", and `LlamaVisionModelInfo` for vision models appending "-Vision" suffix. Creates four ModelMeta instances: (1) `LlamaMeta_3_1` - Llama 3.1 8B with optional Instruct tag, (2) `LlamaMeta_3_2_Base` - Llama 3.2 base models (1B, 3B), (3) `LlamaMeta_3_2_Instruct` - instruction-tuned 3.2 models with GGUF support, (4) `LlamaMeta_3_2_Vision` - multimodal vision models (11B, 90B) with size-specific quantization (11B supports BNB/UNSLOTH, 90B only NONE). All from org="meta-llama".

**Significance:** Core support for Meta's flagship Llama models, widely used in the open-source LLM ecosystem. The distinction between 3.1 and 3.2 versions, plus specialized vision models, reflects Meta's model evolution. Vision models enable multimodal applications, while size variations (1B to 90B) serve different compute budgets and use cases.
