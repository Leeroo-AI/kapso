# File: `tests/saving/text_to_speech_models/test_whisper.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 195 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration test for Whisper speech-to-text model with LoRA fine-tuning, saving/merging, and inference validation.

**Mechanism:** The test follows a 7-section workflow: (1) loads Whisper model with FastModel wrapper and applies LoRA adapters targeting q_proj and v_proj layers, (2) verifies model is PeftModel instance, (3) validates base model class preservation through LoRA wrapper, (4) saves and merges LoRA weights into base model using save_pretrained_merged, (5) reloads merged model from disk, (6) downloads sample FLAC audio file from Wikimedia Commons, and (7) runs automatic speech recognition pipeline to transcribe audio and validates that expected phrases ("birch canoe slid on smooth planks", "sheet to dark blue background", etc.) appear in transcription output. Includes cleanup of temporary directories.

**Significance:** Critical validation test for Whisper model support in Unsloth framework. Ensures that speech-to-text models can be fine-tuned with LoRA, saved/merged correctly without warnings, and perform accurate inference after reload. Tests integration with transformers pipeline API and validates transcription quality against known ground truth phrases.
