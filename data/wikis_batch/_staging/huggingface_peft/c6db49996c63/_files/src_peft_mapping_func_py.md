# File: `src/peft/mapping_func.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 128 |
| Functions | `get_peft_model` |
| Imports | __future__, auto, config, mapping, mixed_model, peft_model, transformers, tuners, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Primary factory function for creating PEFT models from base models and configurations.

**Mechanism:** Takes a base model and PeftConfig, updates the base_model_name_or_path, validates the model state, applies dtype autocasting for adapters, selects the appropriate wrapper class (PeftMixedModel for mixed adapters, task-specific PeftModel subclasses for specific tasks, or base PeftModel otherwise), and returns the wrapped model ready for training or inference.

**Significance:** The main entry point for PEFT model creation. This is the function users call to add PEFT adapters to their models. Handles critical initialization logic including model state validation, adapter dtype management, and proper class selection. Central to the PEFT workflow and used by virtually all PEFT applications.
