# File: `src/peft/tuners/multitask_prompt_tuning/config.py`

**Category:** Configuration

| Property | Value |
|----------|-------|
| Lines | 62 |
| Classes | `MultitaskPromptTuningInit`, `MultitaskPromptTuningConfig` |
| Imports | dataclasses, enum, peft, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Defines configuration classes for Multitask Prompt Tuning, including initialization strategies and hyperparameters for multi-task learning scenarios.

**Mechanism:**

### MultitaskPromptTuningInit (Enum):
Defines five initialization strategies:
- **TEXT**: Initialize prompt embeddings from text
- **RANDOM**: Random initialization
- **AVERAGE_SOURCE_TASKS**: Average prefix/column matrices from source task training
- **EXACT_SOURCE_TASK**: Use specific source task's prefix/column matrices
- **ONLY_SOURCE_SHARED**: Use only shared prompt embeddings from source training

### MultitaskPromptTuningConfig (DataClass):
Extends `PromptTuningConfig` with multitask-specific parameters:

**Core Parameters**:
- **prompt_tuning_init**: Initialization strategy (from enum above)
- **prompt_tuning_init_state_dict_path**: Path to source model state dict for transfer learning
- **prompt_tuning_init_task**: Source task ID for EXACT_SOURCE_TASK mode (default=0)
- **num_ranks**: Number of ranks for low-rank factorization (default=1)
- **num_tasks**: Number of tasks in multitask setup (default=1)

**Validation** (`__post_init__`):
- Sets `peft_type` to `PeftType.MULTITASK_PROMPT_TUNING`
- Inherits validation from parent `PromptTuningConfig`

**Significance:** This configuration enables knowledge transfer between tasks through learned prompt representations. The factorized approach (prefix_task_cols × prefix_task_rows) allows efficient parameter sharing while maintaining task-specific adaptations. The various initialization strategies support different transfer learning scenarios from fully supervised source tasks to few-shot target tasks.

## Key Features

- **Five Init Strategies**: From random to sophisticated transfer learning
- **Low-Rank Factorization**: Efficient parameter representation
- **Source Task Transfer**: Load pre-trained prompt knowledge
- **Multi-Task Support**: Explicit handling of task count and IDs
- **Flexible Initialization**: Supports averaging or exact task selection

## Usage Scenarios

1. **Cold Start** (RANDOM/TEXT): No source tasks available
2. **Transfer Learning** (AVERAGE_SOURCE_TASKS): Leverage multiple source tasks
3. **Task-Specific Transfer** (EXACT_SOURCE_TASK): Target similar to specific source
4. **Shared Knowledge** (ONLY_SOURCE_SHARED): Use only task-agnostic components
