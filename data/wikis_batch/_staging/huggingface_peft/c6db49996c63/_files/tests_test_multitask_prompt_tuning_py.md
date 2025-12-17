# File: `tests/test_multitask_prompt_tuning.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 288 |
| Classes | `TestMultiTaskPromptTuning` |
| Imports | os, peft, pytest, tempfile, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for MultitaskPromptTuning functionality

**Mechanism:** Validates multi-task prompt tuning with task-specific virtual tokens and gating mechanisms. Tests training, generation with task_ids, saving/loading, various initialization methods (random, average, exact, only-shared), and compatibility with different precision modes (fp32, fp16, bf16)

**Significance:** Ensures multi-task prompt tuning works correctly for training models on multiple tasks simultaneously with task-specific prompts and shared representations
