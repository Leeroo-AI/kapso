# File: `src/peft/tuners/prefix_tuning/__init__.py`

**Category:** tuner module initialization

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | peft.utils.register_peft_method, .config, .model |
| Exports | `PrefixEncoder`, `PrefixTuningConfig` |

## Understanding

**Status:** Explored

**Purpose:** Initializes the Prefix Tuning PEFT method module by importing and exporting key components, and registering the method with PEFT's method registry.

**Mechanism:**
- Imports the core Prefix Tuning components:
  - `PrefixTuningConfig`: Configuration class for Prefix Tuning
  - `PrefixEncoder`: Model implementation
- Registers Prefix Tuning with PEFT framework using `register_peft_method()`:
  - Method name: "prefix_tuning"
  - Config class: PrefixTuningConfig
  - Model class: PrefixEncoder
- Exports both classes in `__all__` for public API

**Significance:** Critical module initialization file that integrates Prefix Tuning into the PEFT framework. The registration call enables PEFT to dynamically instantiate Prefix Tuning adapters when users specify `peft_type="PREFIX_TUNING"` in their configs. Prefix Tuning is a popular prompt learning method that prepends trainable continuous vectors to the keys and values of attention layers.
