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

**Purpose:** Register Microsoft Phi-4 models

**Mechanism:** Defines PhiModelInfo for phi-4 naming format, creates separate ModelMeta configs for base (no instruct tag) and mini-instruct variants, both supporting 1B size with standard quantization options (NONE, BNB, UNSLOTH, plus GGUF for instruct).

**Significance:** Enables Unsloth support for Microsoft's compact Phi-4 model family, providing both base and instruction-tuned variants with appropriate quantization formats for efficient fine-tuning of smaller language models.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
