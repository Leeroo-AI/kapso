# Implementation: multitask_prompt_tuning/model.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/multitask_prompt_tuning/model.py`
- **Size**: 120 lines
- **Description**: Multi-task prompt embedding with task-specific transformations

## Overview

MultitaskPromptEmbedding implements the embedding layer that adapts shared prompts for multiple tasks using low-rank task-specific matrices.

## Core Class: MultitaskPromptEmbedding

### Architecture

```python
class MultitaskPromptEmbedding(PromptEmbedding):
    def __init__(self, config, word_embeddings):
        super().__init__(config, word_embeddings)

        total_virtual_tokens = self.num_virtual_tokens * self.num_transformer_submodules

        # Task-specific low-rank matrices
        self.prefix_task_cols = nn.Parameter(
            torch.normal(0, 0.02, (num_tasks, total_virtual_tokens, num_ranks))
        )
        self.prefix_task_rows = nn.Parameter(
            torch.normal(0, 0.02, (num_tasks, num_ranks, token_dim))
        )
```

### Forward Pass

```python
def forward(self, indices, task_ids):
    if task_ids is None:
        raise ValueError("task_ids cannot be None")

    # Get shared embedding
    prompt_embeddings = self.embedding(indices)  # (batch, num_tokens, hidden)

    # Select task-specific matrices
    task_cols = torch.index_select(self.prefix_task_cols, 0, task_ids)  # (batch, tokens, ranks)
    task_rows = torch.index_select(self.prefix_task_rows, 0, task_ids)  # (batch, ranks, hidden)

    # Compute task-specific transformation
    task_prompts = torch.matmul(task_cols, task_rows)  # (batch, tokens, hidden)

    # Element-wise modulation
    prompt_embeddings *= task_prompts

    return prompt_embeddings
```

### Initialization from Source Tasks

**Average Multiple Tasks**:
```python
if config.prompt_tuning_init == MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS:
    state_dict = load(config.prompt_tuning_init_state_dict_path)
    prefix_task_cols_ = state_dict["prefix_task_cols"].mean(0, keepdim=True)
    prefix_task_rows_ = state_dict["prefix_task_rows"].mean(0, keepdim=True)
    self.load_state_dict({
        "embedding.weight": state_dict["prompt_embeddings"],
        "prefix_task_cols": prefix_task_cols_,
        "prefix_task_rows": prefix_task_rows_,
    }, strict=True)
```

**Copy Specific Task**:
```python
elif config.prompt_tuning_init == MultitaskPromptTuningInit.EXACT_SOURCE_TASK:
    task_id = config.prompt_tuning_init_task
    prefix_task_cols_ = state_dict["prefix_task_cols"][task_id, ...].unsqueeze(0)
    prefix_task_rows_ = state_dict["prefix_task_rows"][task_id, ...].unsqueeze(0)
    # ... load ...
```

**Only Shared Embeddings**:
```python
elif config.prompt_tuning_init == MultitaskPromptTuningInit.ONLY_SOURCE_SHARED:
    self.load_state_dict({
        "embedding.weight": state_dict["prompt_embeddings"]
    }, strict=False)  # Don't load task-specific parts
```

## Key Features

### Task-Specific Modulation

**Element-wise multiplication** allows each task to:
- Amplify/suppress certain prompt dimensions
- Learn task-specific patterns
- Share common knowledge via base embedding

### Low-Rank Efficiency

**Parameter Count**:
- Full per-task prompts: T × V × d
- Low-rank multi-task: V × d + T × (V × r + r × d)
- **Savings**: Significant when r << d and T is large

**Example** (V=20, d=768, T=10, r=8):
- Full: 10 × 20 × 768 = 153,600
- Low-rank: 15,360 + 10 × (160 + 6,144) = 78,400 (49% reduction)

## Usage Example

```python
from peft import get_peft_model, MultitaskPromptTuningConfig, TaskType

config = MultitaskPromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=20,
    num_tasks=10,
    num_ranks=8,
    prompt_tuning_init="RANDOM"
)

model = get_peft_model(base_model, config)

# Training
for batch in dataloader:
    # Must provide task_ids
    outputs = model(
        input_ids=batch["input_ids"],
        task_ids=batch["task_ids"],  # Required!
        labels=batch["labels"]
    )
    loss = outputs.loss
    loss.backward()
```

## Cross-References

- **Config**: `multitask_prompt_tuning/config.py`
- **Base Class**: `peft.tuners.prompt_tuning.PromptEmbedding`
- **Paper**: [Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning](https://huggingface.co/papers/2303.02861)
