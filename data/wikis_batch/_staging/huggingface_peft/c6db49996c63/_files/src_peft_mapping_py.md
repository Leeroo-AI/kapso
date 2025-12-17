# File: `src/peft/mapping.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 92 |
| Functions | `get_peft_config`, `inject_adapter_in_model` |
| Imports | __future__, torch, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Maintains registry mappings from PEFT types to their corresponding config and tuner classes, and provides adapter injection functionality.

**Mechanism:** Defines four dictionaries (PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING, PEFT_TYPE_TO_MIXED_MODEL_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING) filled by tuner registration. get_peft_config instantiates configs from dicts. inject_adapter_in_model adds PEFT layers directly to a model without PeftModel wrapper.

**Significance:** Central registry system that enables dynamic discovery and instantiation of PEFT methods. Critical for the library's extensibility - new PEFT methods register themselves by populating these mappings. The inject_adapter_in_model function is key for integrations that need PEFT adapters without the full model wrapper.
