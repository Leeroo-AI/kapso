# File: `src/peft/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 250 |
| Imports | auto, config, mapping, mapping_func, mixed_model, peft_model, tuners, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization module that exports the public API of the PEFT library.

**Mechanism:** Imports and re-exports all major classes, functions, and constants from submodules including AutoPeft models, PeftModel classes, all tuner configurations (LoRA, Prefix Tuning, P-Tuning, etc.), utility functions, and type definitions. Sets the package version number and defines __all__ for explicit export control.

**Significance:** Central entry point for the PEFT library that provides a unified interface for users. Makes all PEFT functionality accessible from a single import statement (from peft import *). Essential for maintaining a clean and consistent public API across the entire library.
