# File: `src/peft/tuners/p_tuning/__init__.py`

**Category:** tuner module initialization

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | peft.utils.register_peft_method, .config, .model |
| Exports | `PromptEncoder`, `PromptEncoderConfig`, `PromptEncoderReparameterizationType` |

## Understanding

**Status:** Explored

**Purpose:** Initializes the P-Tuning PEFT method module by importing and exporting key components, and registering the method with PEFT's method registry.

**Mechanism:**
- Imports the core P-Tuning components:
  - `PromptEncoderConfig`: Configuration class for P-Tuning
  - `PromptEncoderReparameterizationType`: Enum for encoder types (MLP/LSTM)
  - `PromptEncoder`: Model implementation
- Registers P-Tuning with PEFT framework using `register_peft_method()`:
  - Method name: "p_tuning"
  - Config class: PromptEncoderConfig
  - Model class: PromptEncoder
- Exports all three classes in `__all__` for public API

**Significance:** Critical module initialization file that integrates P-Tuning into the PEFT framework. The registration call enables PEFT to dynamically instantiate P-Tuning adapters when users specify `peft_type="P_TUNING"` in their configs. This follows PEFT's plugin architecture where each tuning method is a self-contained module.
