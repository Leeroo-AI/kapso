# File: `tests/saving/gpt-oss-merge/train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 102 |
| Functions | `safe_remove_directory`, `formatting_prompts_func` |
| Imports | datasets, gc, os, shutil, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the complete training-to-deployment workflow for GPT models by performing QLoRA training, merging adapters, and saving the final model in production-ready format.

**Mechanism:** Loads a GPT model with Unsloth, applies LoRA adapters, trains using SFTTrainer on formatted instruction datasets, merges trained adapters into base weights, and saves the merged model to disk with proper configuration files.

**Significance:** Validates the end-to-end pipeline from training to model persistence, ensuring Unsloth-trained GPT models can be properly saved and distributed for downstream applications without requiring the original training infrastructure.
