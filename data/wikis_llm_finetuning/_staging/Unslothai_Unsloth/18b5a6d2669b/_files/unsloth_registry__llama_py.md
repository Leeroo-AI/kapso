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

**Purpose:** Registers Meta LLaMA model variants across versions 3.1 and 3.2, including text-only and vision-capable models.

**Mechanism:** Defines LlamaModelInfo for standard text models using "Llama-{version}-{size}B" naming, and LlamaVisionModelInfo appending "-Vision" suffix. Creates four ModelMeta instances: LlamaMeta_3_1 (8B, base/Instruct tags, none/bnb/unsloth quants), LlamaMeta_3_2_Base (1B/3B, no instruct, none/bnb/unsloth), LlamaMeta_3_2_Instruct (1B/3B with Instruct tag, adds gguf quant), and LlamaMeta_3_2_Vision (11B/90B multimodal, base/Instruct tags, size-dependent quants where 11B gets bnb/unsloth but 90B only gets none). Provides separate registration functions for 3.1, 3.2 text, and 3.2 vision, coordinated by register_llama_models().

**Significance:** Manages Meta's flagship LLaMA family including the transition from 3.1 to 3.2 and the introduction of vision capabilities, with differentiated quantization support based on model size reflecting practical memory and performance constraints.
