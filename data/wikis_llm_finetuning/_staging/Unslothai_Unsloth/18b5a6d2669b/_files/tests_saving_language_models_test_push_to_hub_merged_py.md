# File: `tests/saving/language_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 204 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the push_to_hub_merged functionality for language models by training a LoRA adapter on Llama-3.2-1B-Instruct and validating Hub upload/download workflow.

**Mechanism:** Loads a 4-bit quantized model, applies LoRA adapters with rank 16, trains on openassistant-guanaco-reformatted dataset for 30 steps using SFTTrainer with response-only training, then uploads the merged model to Hugging Face Hub using push_to_hub_merged and verifies successful download.

**Significance:** Critical integration test that validates the complete model save-merge-upload-download pipeline for language models, ensuring merged models can be successfully shared and retrieved from Hugging Face Hub.
