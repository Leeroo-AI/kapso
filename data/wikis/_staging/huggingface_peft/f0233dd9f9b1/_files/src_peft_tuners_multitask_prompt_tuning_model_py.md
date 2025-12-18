# File: `src/peft/tuners/multitask_prompt_tuning/model.py`

**Category:** Model Implementation

| Property | Value |
|----------|-------|
| Lines | 120 |
| Classes | `MultitaskPromptEmbedding` |
| Imports | config, peft, torch |

## Understanding

**Status:** Fully explored

**Purpose:** Implements `MultitaskPromptEmbedding` class that creates task-specific soft prompts using low-rank factorization for efficient multi-task learning.

**Mechanism:**

### Initialization:
1. Inherits from `PromptEmbedding` base class
2. Stores multitask-specific parameters: `num_tasks`, `num_ranks`, `num_virtual_tokens`
3. Determines `num_transformer_submodules` (2 for seq2seq, 1 otherwise)

### Learnable Parameters:
- **prefix_task_cols**: `(num_tasks, total_virtual_tokens, num_ranks)`
  - Task-specific column matrices
  - Initialized with normal(0, 0.02)
- **prefix_task_rows**: `(num_tasks, num_ranks, token_dim)`
  - Task-specific row matrices
  - Initialized with normal(0, 0.02)
- **embedding.weight**: Shared prompt embeddings across tasks

### Initialization from Source Tasks:
Supports three sophisticated initialization modes:

**1. AVERAGE_SOURCE_TASKS**:
```python
prefix_task_cols = mean(source_cols, dim=0)  # Average across tasks
prefix_task_rows = mean(source_rows, dim=0)
```

**2. EXACT_SOURCE_TASK**:
```python
# Select specific task
prefix_task_cols = source_cols[task_id]
prefix_task_rows = source_rows[task_id]
```

**3. ONLY_SOURCE_SHARED**:
```python
# Only load shared embeddings, random init for task-specific
embedding.weight = source_embeddings
```

### Forward Pass:
```python
def forward(self, indices, task_ids):
    # Get shared embeddings
    prompt_embeddings = self.embedding(indices)

    # Get task-specific factors
    task_cols = index_select(prefix_task_cols, 0, task_ids)
    task_rows = index_select(prefix_task_rows, 0, task_ids)

    # Low-rank task prompts
    task_prompts = task_cols @ task_rows

    # Element-wise multiplication
    return prompt_embeddings * task_prompts
```

**Significance:** This implements the core idea from the MIT-IBM Watson paper (https://huggingface.co/papers/2303.02861): learning task-specific prompts through low-rank factorization. The multiplication of shared embeddings with task-specific factors enables:

1. **Parameter Efficiency**: Low-rank factorization reduces parameters vs full task-specific prompts
2. **Knowledge Sharing**: Shared embeddings capture task-agnostic knowledge
3. **Task Specialization**: Task-specific factors adapt prompts to each task
4. **Transfer Learning**: Initialize from source tasks for few-shot target tasks
5. **Scalability**: Add new tasks without retraining everything

The element-wise multiplication allows the shared prompt to be modulated by task-specific transformations, creating effective task-conditional prompts.

## Key Technical Details

- **Low-Rank Structure**: Factorizes task prompts as `cols × rows`
- **Element-wise Modulation**: Multiplies shared × task-specific
- **Task Indexing**: Requires task_ids in forward pass
- **Flexible Loading**: Supports .safetensors and torch formats
- **Normal Initialization**: Mean=0, std=0.02 for stability
- **Transformer Submodules**: Handles encoder-only and encoder-decoder models

## References

- Paper: https://huggingface.co/papers/2303.02861
- Institution: MIT-IBM Watson Research Lab
- Method: Low-rank factorization for multitask prompts
