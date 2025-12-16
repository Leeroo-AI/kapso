# File: `unsloth/trainer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `UnslothTrainingArguments`, `UnslothTrainer` |
| Imports | dataclasses, functools, inspect, logging, os, transformers, trl, typing, unsloth, unsloth_zoo, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extends TRL's SFTTrainer with Unsloth-specific optimizations including automatic padding-free training, sample packing, custom embedding learning rates, and backward compatibility patches for TRL API changes.

**Mechanism:**
- **UnslothTrainingArguments**: Extends TrainingArguments/SFTConfig with `embedding_learning_rate` parameter for differential learning rates on embedding layers
- **UnslothTrainer**: Extends SFTTrainer with custom optimizer creation:
  - Separates parameters into "embeddings" (modules_to_save.default.weight) and "non_embeddings"
  - Creates optimizer with different learning rates per group (typically 5e-5 for embeddings, 2e-4 for others)
  - Prevents embedding layers from overfitting by using lower learning rate
- **Gradient accumulation fix** (`unsloth_train`):
  - For transformers <= 4.45.2, uses custom `_unsloth_train()` to fix accumulation bugs
  - For newer versions, delegates to standard `trainer.train()`
- **Auto padding-free training** (`_patch_sft_trainer_auto_packing`):
  - Detects when padding-free/packing is beneficial (non-VLM, no custom collator)
  - Configures via `configure_padding_free()` or `configure_sample_packing()`
  - Enables via `enable_padding_free_metadata()` or `enable_sample_packing()`
  - Skips for blocklisted models (gemma2, gpt_oss) that don't support padding-free
  - Automatically disables for vision models (ProcessorMixin detected)
  - Gracefully falls back if auto-packing fails
- **TRL backward compatibility** (`_patch_trl_trainer`):
  - Patches all TRL trainers (SFT, DPO, GRPO, KTO, etc.) to accept legacy `tokenizer=` parameter
  - Maps `tokenizer` to `processing_class` for TRL >= 0.13.0
  - Migrates parameters from TrainingArguments to trainer-specific Config classes
  - Reconstructs kwargs to match new TRL API while maintaining old API compatibility
- **Sample packing**: Bins sequences by length to minimize padding waste, enables >2x training speedup

**Significance:** This module makes Unsloth's 2x speed improvements actually work in practice. Padding-free training eliminates computation on padding tokens (often 30-50% of tokens in a batch). Sample packing further reduces waste by grouping similar-length sequences. The automatic detection ensures users get these benefits without configuration. The backward compatibility patches prevent breaking changes when TRL updates its API, maintaining a stable user experience across versions. Differential embedding learning rates prevent overfitting when extending vocabulary with new tokens.
