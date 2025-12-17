# File: `src/peft/tuners/multitask_prompt_tuning/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 62 |
| Classes | `MultitaskPromptTuningInit`, `MultitaskPromptTuningConfig` |
| Imports | dataclasses, enum, peft, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** Configuration for Multitask Prompt Tuning, defining initialization strategies and hyperparameters for shared prompt embeddings across multiple tasks.

**Mechanism:** Extends PromptTuningConfig with MultitaskPromptTuningInit enum (TEXT, RANDOM, AVERAGE_SOURCE_TASKS, EXACT_SOURCE_TASK, ONLY_SOURCE_SHARED) to control initialization from source tasks. Additional fields include prompt_tuning_init_state_dict_path for loading pretrained source prompts, prompt_tuning_init_task for selecting specific source tasks, num_ranks for decomposition rank, and num_tasks for the number of tasks.

**Significance:** Enables transfer learning and multi-task learning with prompt tuning. The various initialization modes allow leveraging knowledge from pretrained source tasks to improve downstream task performance. The decomposed structure (prefix_task_cols × prefix_task_rows) enables efficient parameter sharing across tasks.
