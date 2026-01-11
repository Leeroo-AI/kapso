# File: `tests/saving/text_to_speech_models/test_csm.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 168 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests save-merge-reload-inference pipeline for CSM (text-to-speech) models, validating LoRA adapter integration and audio generation capability after merging.

**Mechanism:** Loads unsloth/csm-1b with CsmForConditionalGeneration, applies LoRA adapters (rank 32) to attention and MLP layers, verifies PeftModel wrapping and base model class preservation, saves with save_pretrained_merged, reloads the merged model, and generates audio output using the processor with configurable temperature and top-p parameters.

**Significance:** Validates Unsloth's support for specialized text-to-speech architectures, ensuring LoRA merging works correctly for generative audio models and produces functional TTS output after save-reload cycle.
