# File: `tests/test_decoder_models.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1001 |
| Classes | `TestDecoderModels` |
| Imports | accelerate, json, peft, platform, pytest, safetensors, tempfile, testing_common, testing_utils, torch, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for decoder-only models (GPT, OPT, Llama, etc.) with PEFT.

**Mechanism:** Contains `TestDecoderModels` class inheriting from `PeftCommonTester` with 30+ parametrized tests across 6 models and 20+ PEFT configs. Tests cover: model attributes, training (standard and gradient checkpointing), inference (generation, half precision), adapter management (save/load, merge, delete, weighted combination), layer indexing, prompt learning methods, device mapping, and special cases (LoRA layer replication, prefix tuning with GQA, embedding scaling for Gemma3).

**Significance:** Provides comprehensive validation of PEFT methods on decoder-only architectures, ensuring all adapter types work correctly across different model families and handle edge cases like grouped query attention and embedding scaling.
