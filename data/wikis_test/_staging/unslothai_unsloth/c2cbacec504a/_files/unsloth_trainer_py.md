# File: `unsloth/trainer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `UnslothTrainingArguments`, `UnslothTrainer` |
| Imports | dataclasses, functools, inspect, logging, os, transformers, trl, typing, unsloth, unsloth_zoo, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides custom trainer classes extending TRL's SFTTrainer with support for sample packing, padding-free batching, embedding learning rates, and backwards compatibility with different TRL versions.

**Mechanism:** Defines UnslothTrainer extending SFTTrainer with custom optimizer creation supporting separate learning rates for embeddings. Implements _patch_sft_trainer_auto_packing() to auto-enable sample packing and padding-free training. Patches trainer __init__ to handle version differences between TRL 0.11 and 0.13+. Wraps trainer initialization to translate old "tokenizer" parameter to new "processing_class".

**Significance:** Makes advanced training techniques like padding-free batching and sample packing automatic and transparent to users. Maintains backward compatibility across multiple TRL versions so users don't face breaking changes.
