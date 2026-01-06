# File: `tests/test_decoder_models.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1001 |
| Classes | `TestDecoderModels` |
| Imports | accelerate, json, peft, platform, pytest, safetensors, tempfile, testing_common, testing_utils, torch, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for decoder-only language models

**Mechanism:** Tests PEFT adapters (LoRA, AdaLora, BOFT, IA3, PromptTuning, etc.) on decoder models like GPT-2, OPT, GPT-J, Llama, Qwen2, and Gemma3. Tests common operations including training with Trainer, CPT adapter functionality, and model-specific behaviors

**Significance:** Test coverage for decoder model architectures with all PEFT methods
