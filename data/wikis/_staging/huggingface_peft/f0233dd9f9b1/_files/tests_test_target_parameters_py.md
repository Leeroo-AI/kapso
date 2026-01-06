# File: `tests/test_target_parameters.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 546 |
| Classes | `MyAutoModelForCausalLM`, `TestDecoderModelsTargetParameters`, `TestTargetParameters`, `MyLinear`, `MyModule`, `MyParametrization` |
| Imports | peft, pytest, testing_common, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for target_parameters feature

**Mechanism:** Tests targeting specific parameters (like MoE expert layers in Llama4/GptOss) using target_parameters config instead of target_modules, including cross-module targeting

**Significance:** Test coverage for fine-grained parameter targeting
