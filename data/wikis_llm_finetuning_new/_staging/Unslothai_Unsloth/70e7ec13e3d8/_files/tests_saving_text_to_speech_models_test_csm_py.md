# File: `tests/saving/text_to_speech_models/test_csm.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 168 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** Explored

**Purpose:** Tests the complete workflow of fine-tuning, saving, merging, loading, and running inference on the CSM (Conversational Speech Model) text-to-speech model.

**Mechanism:** The test loads the CSM-1B model using FastModel with CsmForConditionalGeneration as the auto_model. It applies LoRA adapters (r=32) to attention and MLP layers, then validates the model is a PeftModel instance and that the underlying base model class is correct. The model is saved using save_pretrained_merged() with warnings treated as errors to catch any issues. After reloading from disk, it runs TTS inference with customizable generation parameters (depth_decoder_temperature, top_k, top_p, etc.) to generate a WAV audio file at 24kHz sample rate using soundfile. The test verifies each section passes without exceptions.

**Significance:** Validates Unsloth's support for text-to-speech models beyond traditional language models. This ensures that the LoRA fine-tuning and merging workflow works correctly for audio generation models, expanding Unsloth's multimodal capabilities.
