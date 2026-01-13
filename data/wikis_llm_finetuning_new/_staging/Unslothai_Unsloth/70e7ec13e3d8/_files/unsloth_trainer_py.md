# File: `unsloth/trainer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 438 |
| Classes | `UnslothTrainingArguments`, `UnslothTrainer` |
| Imports | dataclasses, functools, inspect, logging, os, psutil, transformers, trl, typing, unsloth, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Custom trainer classes extending TRL's SFTTrainer with Unsloth-specific optimizations including embedding learning rate support, gradient accumulation fixes, and automatic sample packing.

**Mechanism:** `UnslothTrainingArguments` extends `TrainingArguments` to add `embedding_learning_rate` parameter. `UnslothTrainer` extends `SFTTrainer` with `_create_unsloth_optimizer()` for separate embedding/non-embedding learning rates. Helper functions `_backwards_compatible_trainer()` and `_patch_trl_trainer()` ensure compatibility with TRL versions 0.13.0+. `_patch_sft_trainer_auto_packing()` enables automatic sample packing for efficient batching.

**Significance:** Core component - the main training interface for Unsloth. Provides the optimized training loop that delivers Unsloth's speed and memory efficiency improvements over standard HuggingFace training.
