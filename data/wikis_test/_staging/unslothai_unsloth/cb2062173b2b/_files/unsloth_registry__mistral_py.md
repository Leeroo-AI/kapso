# File: `unsloth/registry/_mistral.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 88 |
| Classes | `MistralSmallModelInfo` |
| Functions | `register_mistral_small_models`, `register_mistral_models` |
| Imports | copy, unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers Mistral AI's Small model variants (24B parameter models) across multiple release versions with custom naming conventions.

**Mechanism:** Defines `MistralSmallModelInfo` with version-specific naming logic: version "2503" uses format "Mistral-3.1-24B-{tag}-2503", while other versions use "Mistral-24B-{tag}-{version}". Creates four ModelMeta instances using `copy.deepcopy` for efficient variant creation: Base and Instruct variants for both "2503" (March 2025) and "2501" (January 2025) releases. Base models support NONE/UNSLOTH/BNB quantization; Instruct models add GGUF support. Version "2409" (September 2024) is defined but noted as not uploaded to Unsloth. All are text-only models from org="mistralai".

**Significance:** Supports Mistral AI's Small model family with 24B parameters, tracking multiple release versions over time. The version-based registry allows users to access specific model checkpoints with known characteristics. The use of deepcopy for variant creation demonstrates efficient metadata management when models share most attributes but differ in tags and quantization support.
