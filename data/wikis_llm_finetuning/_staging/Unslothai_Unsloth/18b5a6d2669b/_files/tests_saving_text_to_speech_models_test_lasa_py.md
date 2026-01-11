# File: `tests/saving/text_to_speech_models/test_lasa.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 220 |
| Functions | `find_lora_base_model`, `ids_to_speech_tokens`, `extract_speech_ids` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests LASA (Llama-based Speech Architecture) TTS model with LoRA merging and speech token generation using XCodec2 audio codec for decoding.

**Mechanism:** Loads Llasa-1B model with LoRA adapters (rank 128 on q_proj/v_proj), saves merged model, reloads it, generates speech by processing text through special tokens (TEXT_UNDERSTANDING_START/END, SPEECH_GENERATION_START/END), extracts speech IDs from generated tokens, and decodes to audio waveform using XCodec2Model codec at 16kHz.

**Significance:** Validates Unsloth's support for Llama-based speech synthesis models with specialized tokenization schemes, ensuring the merge process preserves the model's ability to generate proper speech tokens compatible with external audio codecs.
