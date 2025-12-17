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

**Purpose:** Demonstrates offline inference with audio language models.

**Mechanism:** Provides model-specific configuration functions (12 models including AudioFlamingo3, Gemma3N, Granite Speech, MiniCPM-O, Phi-4-multimodal, Qwen2-Audio, Ultravox, Voxtral, Whisper) that return ModelRequestData with appropriate engine args, prompts with audio placeholders, and optional LoRA requests. Uses AudioAsset to load example audio files and runs batch inference.

**Significance:** Comprehensive example showcasing correct prompt formats and configurations for various audio language models supported by vLLM.
