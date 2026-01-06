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

**Purpose:** Tests for ARROW (Adaptive Rank-Optimized Weight Routing) adapter

**Mechanism:** Tests ARROW routing functionality including adapter creation, rank compatibility checking, task-specific vs general adapters, training behavior, and adapter source resolution across different models

**Significance:** Test coverage for ARROW adapter routing mechanism
