# File: `tests/saving/text_to_speech_models/test_whisper.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 195 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** End-to-end test for Whisper speech-to-text model fine-tuning, saving, merging, and inference within the Unsloth framework.

**Mechanism:** The test follows a comprehensive pipeline: (1) loads a Whisper-large-v3 model using FastModel with auto-detection for the WhisperForConditionalGeneration class, (2) applies LoRA adapters targeting q_proj and v_proj with rank 64 and Unsloth gradient checkpointing, (3) validates the model is a PeftModel wrapping the correct base model class, (4) saves and merges the model using save_pretrained_merged while catching warnings as errors, (5) reloads the merged model from disk, (6) downloads a sample audio file from Wikimedia, (7) runs inference using a transformers pipeline with the merged model, and (8) asserts the transcription contains expected phrases. Cleanup removes the saved model and compiled cache directories.

**Significance:** This test validates the complete workflow for audio/speech models in Unsloth, ensuring that Whisper models can be fine-tuned with LoRA adapters and properly saved/merged without warnings. It verifies cross-modal model support beyond text-only language models and confirms inference accuracy post-merge.
