# File: `tests/test_multitask_prompt_tuning.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 288 |
| Classes | `TestMultiTaskPromptTuning` |
| Imports | os, peft, pytest, tempfile, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for MultitaskPromptTuning adapter

**Mechanism:** Tests MultitaskPromptTuning configuration and functionality including training preparation, int8 support, save/load operations, and task-specific prompt behavior on Llama models

**Significance:** Test coverage for multitask prompt tuning PEFT method
