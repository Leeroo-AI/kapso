# File: `src/peft/mixed_model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 473 |
| Classes | `PeftMixedModel` |
| Imports | __future__, accelerate, config, contextlib, os, peft, peft_model, torch, transformers, tuners, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Specialized PEFT model class that supports mixing different types of adapters (e.g., LoRA + IA3) in a single model.

**Mechanism:** Wraps a base model with the MixedModel tuner that can handle multiple adapter types. Provides methods to add, load, delete, and activate adapters dynamically. Manages adapter compatibility checks and forward/generate calls that apply multiple active adapters. Currently doesn't support saving (only loading and inference).

**Significance:** Enables advanced use cases where different adapter types are combined for improved performance or flexibility. Important for research and experimentation with adapter composition. The limitation to inference-only reflects the complexity of training mixed adapters. Extends PEFT's capabilities beyond single-adapter-type models.
