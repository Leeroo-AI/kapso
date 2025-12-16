# File: `unsloth/trainer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `UnslothTrainingArguments`, `UnslothTrainer` |
| Imports | dataclasses, functools, inspect, logging, os, transformers, trl, typing, unsloth, unsloth_zoo, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Custom training arguments and trainer enhancement

**Mechanism:** Extends HuggingFace's SFTTrainer with Unsloth-specific optimizations including auto-packing, padding-free training, embedding learning rates, and backwards compatibility. Patches TRL trainers dynamically to work with newer transformers versions, handles model type checking to disable incompatible optimizations (e.g., for VLMs), and provides custom optimizer creation for differential learning rates on embeddings.

**Significance:** Makes Unsloth's training features (2x faster training, lower VRAM) accessible through familiar HuggingFace training APIs. Auto-enables sample packing and padding-free training when applicable, with smart fallback when incompatibilities are detected. Ensures smooth transitions between different TRL/transformers versions.
