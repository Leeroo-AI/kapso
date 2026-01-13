# File: `tests/saving/text_to_speech_models/test_lasa.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 220 |
| Functions | `find_lora_base_model`, `ids_to_speech_tokens`, `extract_speech_ids` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings, ... +1 more |

## Understanding

**Status:** Explored

**Purpose:** Tests the fine-tuning, saving, merging, and inference workflow for the Llasa (LLM-based Speech Synthesis) text-to-speech model using XCodec2 for audio decoding.

**Mechanism:** The test loads Llasa-1B using FastLanguageModel with a high LoRA rank (r=128) targeting q_proj and v_proj. After validating PeftModel type and base model class, it saves the merged model. For inference, it uses a unique token-based approach: text is wrapped in TEXT_UNDERSTANDING markers, the model generates special speech tokens (<|s_N|>), which are extracted using extract_speech_ids() and converted to audio using XCodec2Model.decode_code(). The test requires ffmpeg, soundfile, and xcodec2 packages. Audio output is saved at 16kHz sample rate.

**Significance:** Validates Unsloth's support for LLM-based TTS models that use discrete speech tokens rather than direct audio generation. The Llasa architecture represents a different approach to TTS compared to CSM, and this test ensures both paradigms work with Unsloth's fine-tuning framework.
