# Implementation: multitask_prompt_tuning/config.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/multitask_prompt_tuning/config.py`
- **Size**: 62 lines
- **Description**: Multi-task prompt tuning configuration

## Overview

Multi-task prompt tuning learns task-specific prompt modifications while sharing a common prompt embedding base. It uses low-rank task-specific transformations to adapt prompts efficiently.

## Core Configuration

### MultitaskPromptTuningInit

```python
class MultitaskPromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"                           # Initialize from text
    RANDOM = "RANDOM"                       # Random initialization
    AVERAGE_SOURCE_TASKS = "AVERAGE_SOURCE_TASKS"     # Average source task prompts
    EXACT_SOURCE_TASK = "EXACT_SOURCE_TASK"           # Copy specific source task
    ONLY_SOURCE_SHARED = "ONLY_SOURCE_SHARED"         # Only shared embeddings
```

### MultitaskPromptTuningConfig

```python
@dataclass
class MultitaskPromptTuningConfig(PromptTuningConfig):
    prompt_tuning_init: Union[MultitaskPromptTuningInit, str] = MultitaskPromptTuningInit.RANDOM
    prompt_tuning_init_state_dict_path: Optional[str] = None    # Path to source weights
    prompt_tuning_init_task: Optional[int] = 0                  # Source task ID
    num_ranks: Optional[int] = 1                                # Low-rank dimension
    num_tasks: Optional[int] = 1                                # Number of tasks
```

### Architecture

**Prompt Composition**:
```
task_prompt = shared_embedding * (task_cols @ task_rows)
```

**Parameters**:
- `shared_embedding`: (num_virtual_tokens, hidden_size)
- `task_cols`: (num_tasks, num_virtual_tokens, num_ranks)
- `task_rows`: (num_tasks, num_ranks, hidden_size)

**Total Parameters**: O(T × r × (d + V)) where T=tasks, r=ranks, d=hidden_size, V=virtual_tokens

## Transfer Learning

**Source Task Initialization**:
1. Train on multiple source tasks
2. Save `prefix_task_cols` and `prefix_task_rows`
3. Initialize target task from average or specific source

**Benefit**: Faster convergence on target tasks with related source tasks

## Cross-References

- **Model**: `multitask_prompt_tuning/model.py`
- **Paper**: [Multitask Prompt Tuning](https://huggingface.co/papers/2303.02861)
