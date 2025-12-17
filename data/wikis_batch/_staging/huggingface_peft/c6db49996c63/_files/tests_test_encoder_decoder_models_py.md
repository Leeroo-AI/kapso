# File: `tests/test_encoder_decoder_models.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 434 |
| Classes | `TestEncoderDecoderModels` |
| Imports | peft, pytest, tempfile, testing_common, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for encoder-decoder model PEFT adapters across all supported configurations.

**Mechanism:** Contains the TestEncoderDecoderModels class that inherits from PeftCommonTester and runs parametrized tests on encoder-decoder models (T5, BART) with various PEFT methods including LoRA, Prefix Tuning, Prompt Tuning, AdaLoRA, IA3, BOFT, OFT, and many others. Tests cover training, inference, generation, adapter management, gradient checkpointing, device mapping, merging, and serialization.

**Significance:** Essential validation of PEFT functionality for sequence-to-sequence models. Ensures all adapter types work correctly with encoder-decoder architectures, covering edge cases like mixed adapter batches, beam search generation, and multi-adapter scenarios.
