# File: `unsloth/trainer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 438 |
| Classes | `UnslothTrainingArguments`, `UnslothTrainer` |
| Imports | dataclasses, functools, inspect, logging, os, psutil, transformers, trl, typing, unsloth, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enhanced training infrastructure with automatic performance optimizations

**Mechanism:** Extends SFTTrainer with: custom UnslothTrainingArguments supporting embedding_learning_rate, UnslothTrainer with specialized optimizer creation for differential learning rates, automatic sample packing detection and configuration, padding-free batching auto-enablement (with model-specific blocklists), TRL trainer backward compatibility patches for API changes, and gradient accumulation fixes for older transformers versions

**Significance:** Core training component that transparently enables 2x+ speedups through sample packing and padding-free batching while maintaining compatibility with TRL/transformers across versions, reducing user configuration burden
