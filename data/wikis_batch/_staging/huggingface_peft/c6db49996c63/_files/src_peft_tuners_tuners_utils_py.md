# File: `src/peft/tuners/tuners_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 2041 |
| Classes | `BaseTuner`, `BaseTunerLayer`, `_ExcludedModule` |
| Functions | `onload_layer`, `check_target_module_exists`, `inspect_matched_modules`, `check_adapters_to_merge`, `clone_module`, `replicate_layers`, `set_adapter`, `delete_adapter`, `... +2 more` |
| Imports | __future__, _buffer_dict, abc, accelerate, collections, config, contextlib, copy, dataclasses, os, ... +10 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Foundational base classes and utilities for all PEFT tuner implementations.

**Mechanism:** Defines BaseTuner (model-level adapter management with injection, merging, deletion) and BaseTunerLayer (layer-level adapter operations with enable/disable, set_adapter, weight management). Includes helper functions for target module matching, adapter validation, layer cloning, and device management for offloaded models.

**Significance:** Architectural core of PEFT's adapter system, providing the abstract interface and shared functionality that all tuning methods (LoRA, IA3, etc.) inherit from, ensuring consistent behavior across different PEFT methods and enabling polymorphic adapter operations.
