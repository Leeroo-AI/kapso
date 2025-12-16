# File: `tests/saving/language_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 204 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** End-to-end integration test that validates the push_to_hub_merged functionality for uploading merged LoRA models to Hugging Face Hub and verifying successful download.

**Mechanism:** Fine-tunes Llama-3.2-1B-Instruct for 30 steps with QLoRA on OpenAssistant dataset using Llama 3.1 chat template and train-on-responses-only mode. Collects HF username and token from environment or user input, then uses push_to_hub_merged to upload the merged model. Verifies upload success, tests downloading the model back from the Hub, and tracks success/failure of each stage (upload and download) with detailed reporting. Cleans up directories after completion.

**Significance:** Essential validation test for the Hub integration workflow, ensuring that merged models can be properly uploaded to and retrieved from Hugging Face Hub. This is crucial for users who want to share their fine-tuned models publicly or store them remotely. Tests the complete roundtrip: train → merge → upload → download.
