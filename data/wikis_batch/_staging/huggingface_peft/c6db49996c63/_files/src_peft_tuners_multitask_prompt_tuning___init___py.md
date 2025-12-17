# File: `src/peft/tuners/multitask_prompt_tuning/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 25 |
| Imports | config, model, peft |

## Understanding

**Status:** ✅ Documented

**Purpose:** Package initialization file that registers Multitask Prompt Tuning method with PEFT and exports configuration and model classes.

**Mechanism:** Imports MultitaskPromptTuningConfig, MultitaskPromptTuningInit enum, and MultitaskPromptEmbedding class, then calls register_peft_method() to register "multitask_prompt_tuning" as a valid PEFT method. Based on MIT-IBM Watson Research Lab work.

**Significance:** Entry point for Multitask Prompt Tuning (https://huggingface.co/papers/2303.02861), which extends standard prompt tuning to share knowledge across multiple tasks via decomposed prompt embeddings with task-specific factors.
