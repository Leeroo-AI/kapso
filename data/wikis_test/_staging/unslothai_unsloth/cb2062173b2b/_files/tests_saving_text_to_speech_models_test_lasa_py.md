# File: `tests/saving/text_to_speech_models/test_lasa.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 220 |
| Functions | `find_lora_base_model`, `ids_to_speech_tokens`, `extract_speech_ids` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration test validating the complete workflow for Llasa TTS models using XCodec2 audio codec, including LoRA fine-tuning, model merging, and multi-stage inference with speech token decoding.

**Mechanism:** Loads Llasa-1B with LoRA adapters targeting q_proj and v_proj. Validates model type and base class preservation through find_lora_base_model. Saves and reloads merged model. Inference pipeline: formats input text with TEXT_UNDERSTANDING markers, applies chat template with continued generation, generates speech tokens autoregressively with custom stopping at SPEECH_GENERATION_END token. Helper functions extract_speech_ids and ids_to_speech_tokens convert between token strings ("<|s_23456|>") and integer IDs. Uses XCodec2Model to decode speech token tensors into 16kHz audio waveform. Requires ffmpeg, soundfile, and xcodec2 packages.

**Significance:** Demonstrates Unsloth's support for advanced TTS architectures that use tokenized speech representations rather than direct audio generation. Validates the preservation of complex generation pipelines through model merging. Important for researchers and developers working with neural codec-based speech synthesis systems. Tests the interaction between Unsloth's model management and external codec models.
