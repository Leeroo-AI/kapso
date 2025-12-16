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

**Purpose:** Registers Meta's Llama 3.1 and 3.2 model families including text-only and vision-enabled variants with appropriate quantization configurations.

**Mechanism:** Defines two ModelInfo subclasses: `LlamaModelInfo` for standard models (format "Llama-{version}-{size}B") and `LlamaVisionModelInfo` for vision models (format "Llama-{version}-{size}B-Vision"). Creates four ModelMeta instances: Llama 3.1 (8B, with/without Instruct), Llama 3.2 Base (1B/3B), Llama 3.2 Instruct (1B/3B with GGUF support), and Llama 3.2 Vision (11B/90B multimodal, size-specific quantization where 11B supports NONE/BNB/UNSLOTH and 90B only NONE). Uses singleton pattern with three registration flags.

**Significance:** Core support for Meta's flagship Llama models, particularly important for the 3.2 family which introduced vision capabilities (11B and 90B variants). The size-specific quantization logic reflects practical constraints (90B models are too large for 4-bit quantization in typical setups). Includes verification against Hugging Face Hub in `__main__` block.
