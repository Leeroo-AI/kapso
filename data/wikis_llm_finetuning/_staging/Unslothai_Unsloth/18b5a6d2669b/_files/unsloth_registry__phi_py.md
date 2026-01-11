# File: `unsloth/registry/_phi.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 74 |
| Classes | `PhiModelInfo` |
| Functions | `register_phi_4_models`, `register_phi_4_instruct_models`, `register_phi_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers Microsoft Phi-4 model variants including base and mini-instruct versions.

**Mechanism:** Defines PhiModelInfo class that constructs names as "phi-{version}" plus optional instruct/quant tags. Creates two ModelMeta instances: PhiMeta4 for base models (None instruct tag, size "1", none/bnb/unsloth quants) and PhiInstructMeta4 for instruction-tuned models ("mini-instruct" tag, size "1", adds gguf quant option). Both are non-multimodal. Provides register_phi_4_models() and register_phi_4_instruct_models() functions with global flags to prevent duplicate registration, coordinated by register_phi_models().

**Significance:** Handles Microsoft's Phi-4 model series with a specific "mini-instruct" variant naming convention, representing compact efficient models that fit the pattern of having both base and instruction-tuned versions with standard Unsloth quantization support.
