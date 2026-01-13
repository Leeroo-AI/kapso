# File: `tests/saving/language_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 204 |
| Functions | `formatting_prompts_func` |
| Imports | datasets, gc, huggingface_hub, multiprocessing, os, pandas, pathlib, sys, tests, torch, ... +4 more |

## Understanding

**Status:** Explored

**Purpose:** Tests the complete workflow of training a LoRA model and pushing the merged result to Hugging Face Hub, then downloading and verifying it.

**Mechanism:** The test loads Llama-3.2-1B-Instruct in 4-bit, applies LoRA adapters to attention and MLP layers, and fine-tunes using SFTTrainer with train_on_responses_only for 30 steps on OpenAssistant-Guanaco. It then uses push_to_hub_merged() to upload the merged model to a user's HF repository (requires HF_USER and HF_TOKEN environment variables). The test validates in two stages: (1) successful upload to Hub, and (2) successful download of the model using FastLanguageModel.from_pretrained() from the Hub repository. A final validation report summarizes pass/fail status for each stage.

**Significance:** Ensures that Unsloth's Hub integration works correctly for sharing fine-tuned merged models. This is critical for the user workflow where models are trained locally and then deployed or shared via Hugging Face Hub.
