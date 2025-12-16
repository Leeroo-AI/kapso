# File: `tests/saving/text_to_speech_models/test_csm.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 168 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration test validating the complete lifecycle for CSM (Conversational Speech Model) text-to-speech models: loading, applying LoRA adapters, saving/merging, reloading, and inference.

**Mechanism:** Section 1 loads csm-1b with CsmForConditionalGeneration and applies LoRA adapters. Section 2-3 verify the model is properly wrapped as PeftModel and validate the base model class type using find_lora_base_model helper. Section 4 performs save_pretrained_merged with warning checks. Section 5 reloads the merged model. Section 6 runs text-to-speech inference generating audio from text input "We just finished fine tuning a text to speech model... and it's pretty good!" with configurable generation parameters (temperature, top_k, top_p, depth_decoder settings). Saves generated audio to WAV file using soundfile. Requires ffmpeg and soundfile packages.

**Significance:** Critical test ensuring Unsloth's model saving works correctly for specialized TTS architectures. Validates that CSM models maintain functionality through the merge-save-reload cycle and can successfully generate audio output. Important for users working with speech synthesis applications and demonstrates Unsloth's support beyond standard text language models.
