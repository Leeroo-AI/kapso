# File: `tests/test_encoder_decoder_models.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 434 |
| Classes | `TestEncoderDecoderModels` |
| Imports | peft, pytest, tempfile, testing_common, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for encoder-decoder sequence-to-sequence models

**Mechanism:** Tests PEFT adapters (LoRA, AdaLoraConfig, BOFT, IA3, PromptTuning, PrefixTuning, etc.) on encoder-decoder models like T5 and BART, covering SEQ_2_SEQ_LM and TOKEN_CLS tasks

**Significance:** Test coverage for encoder-decoder architectures with PEFT methods
