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

**Purpose:** Registers Mistral Small model variants across two versions (2503, 2501) with base and instruction-tuned configurations.

**Mechanism:** Implements MistralSmallModelInfo for version-aware naming (3.1 prefix for 2503 version), uses copy.deepcopy to create instruction variants from base templates, applies uniform quantization support.

**Significance:** Enables support for multiple Mistral Small releases with version-specific naming conventions and quantization differentiation.
