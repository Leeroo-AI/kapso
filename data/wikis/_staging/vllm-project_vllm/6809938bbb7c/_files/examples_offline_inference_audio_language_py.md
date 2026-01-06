# File: `examples/offline_inference/audio_language.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 540 |
| Classes | `ModelRequestData` |
| Functions | `run_audioflamingo3`, `run_gemma3n`, `run_granite_speech`, `run_midashenglm`, `run_minicpmo`, `run_phi4mm`, `run_phi4_multimodal`, `run_qwen2_audio`, `... +6 more` |
| Imports | dataclasses, huggingface_hub, os, transformers, typing, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive example demonstrating audio language models with correct prompt formats for various audio-capable models.

**Mechanism:** Provides model-specific configuration functions for 12+ audio language models (AudioFlamingo3, Gemma3N, Granite Speech, MiDashengLM, MiniCPM-O, Phi-4-multimodal, Qwen2-Audio, Qwen2.5-Omni, Ultravox, Voxtral, Whisper). Each function returns proper EngineArgs, prompts with audio placeholders, and optional LoRA requests tailored to that model's architecture.

**Significance:** Critical reference guide for using audio inputs with vLLM across diverse model architectures. Shows model-specific prompt formatting, audio token placement, and multimodal inference patterns for audio language models.
