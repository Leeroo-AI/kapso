# File: `tests/test_arrow.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 509 |
| Classes | `TestArrowRouting` |
| Functions | `workdir`, `ts_adapters`, `gen_adapter`, `test_training_updates_when_task_adapter_active`, `test_resolve_adapter_source_variants` |
| Imports | copy, os, pathlib, peft, pytest, tests, torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for Arrow (Adaptive Rank-reduced Output Weighting) routing functionality.

**Mechanism:** Contains `TestArrowRouting` class and standalone tests validating: incompatible rank handling, expert count variations, GenKnowSub (General Knowledge Substitution) integration, adapter loading/switching, prototype computation caching, training updates, adapter source resolution (local paths vs Hub repos), merge/unmerge restrictions, Conv2d rejection, and float16 inference. Uses fixtures for creating temporary adapters.

**Significance:** Ensures the Arrow routing mechanism correctly combines multiple task-specific LoRA adapters with optional general knowledge adapters, validating the routing logic and preventing common configuration errors.
