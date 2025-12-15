# File: `tests/saving/language_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 204 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests HuggingFace Hub upload for merged models

**Mechanism:** Trains Llama-3.2-1B with QLoRA for 30 steps, uses push_to_hub_merged to upload merged model to HuggingFace Hub, validates upload succeeded, then tests downloading the model back to verify round-trip works

**Significance:** Validates the push_to_hub_merged functionality enables sharing fine-tuned models, critical for model deployment and collaboration workflows in production environments

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
