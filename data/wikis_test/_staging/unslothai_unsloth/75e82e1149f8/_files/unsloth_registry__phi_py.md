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

**Purpose:** Registers Microsoft's Phi-4 model family including base and mini-instruct variants at 1B parameter size.

**Mechanism:** Defines `PhiModelInfo` class that overrides `construct_model_name()` to format names as "phi-4" (excluding size from name unlike other model families). Creates two ModelMeta instances: `PhiMeta4` for base models (instruct_tags=[None]) and `PhiInstructMeta4` for instruction-tuned models (instruct_tags=["mini-instruct"]), both at 1B size from Microsoft organization. Base models support NONE, BNB, and UNSLOTH quantization; instruct models additionally support GGUF. Uses singleton pattern with two registration flags.

**Significance:** Supports Microsoft's compact yet capable Phi-4 models, particularly the "mini-instruct" variant optimized for efficiency. The simplified naming convention (no size in name) reflects Phi's single-size offering. Being 1B parameter models, these are ideal for resource-constrained deployments. Includes verification script against Hugging Face Hub in `__main__` block.
