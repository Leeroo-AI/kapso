# File: `unsloth/trainer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `UnslothTrainingArguments`, `UnslothTrainer` |
| Imports | dataclasses, functools, inspect, logging, os, transformers, trl, typing, unsloth, unsloth_zoo, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Custom trainer classes and patches for optimized training with Unsloth.

**Mechanism:**
- `UnslothTrainingArguments`: Extends SFTConfig/TrainingArguments with `embedding_learning_rate` parameter
- `UnslothTrainer`: Extends SFTTrainer with custom optimizer supporting separate learning rates for embeddings vs other parameters
- `_create_unsloth_optimizer()`: Creates optimizer with parameter groups for embeddings (default 5e-5) and non-embeddings
- `_backwards_compatible_trainer()`: Decorator that patches TRL trainers for API compatibility across versions (handles tokenizer->processing_class rename, args->config migration)
- `_patch_sft_trainer_auto_packing()`: Automatically enables sample packing or padding-free training when appropriate
- Auto-disables packing for VLMs, custom collators, and blocklisted models (gemma2, gpt_oss)
- `unsloth_train()`: Wrapper that uses custom gradient accumulation fix for transformers <= 4.45.2

**Significance:** Core training infrastructure. The automatic packing/padding-free detection and backwards compatibility patches make Unsloth work seamlessly across TRL versions while providing 2x+ speedups.
