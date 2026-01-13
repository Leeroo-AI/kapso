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

**Purpose:** Defines model metadata and registration logic for Mistral AI's Mistral Small models across different release versions (2501, 2503).

**Mechanism:** Provides `MistralSmallModelInfo` with custom naming logic that handles version-specific naming: 2503 uses `Mistral-Small-3.1-{size}B-{tag}-{version}` while earlier versions use `Mistral-Small-{size}B-{tag}-{version}`. Uses `copy.deepcopy()` to create variant configurations from a base template, modifying version and instruct tags. Registers both base and instruct variants for each version (2501, 2503).

**Significance:** Demonstrates the registry's flexibility in handling vendor-specific naming conventions that change across versions. The use of deepcopy for configuration variants shows a practical pattern for managing closely related model definitions without code duplication.
