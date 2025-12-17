# File: `src/peft/tuners/multitask_prompt_tuning/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 120 |
| Classes | `MultitaskPromptEmbedding` |
| Imports | config, peft, torch |

## Understanding

**Status:** ✅ Documented

**Purpose:** Implements multitask prompt embeddings using low-rank decomposition (prefix_task_cols × prefix_task_rows) to share knowledge across tasks while maintaining task-specific adaptations.

**Mechanism:** Extends PromptEmbedding with trainable prefix_task_cols (num_tasks × total_virtual_tokens × num_ranks) and prefix_task_rows (num_tasks × num_ranks × token_dim) parameters. During forward pass, selects task-specific factors using task_ids, computes task_prompts = matmul(task_cols, task_rows), and applies element-wise multiplication with base prompt embeddings. Supports initialization from pretrained source tasks via state dict loading with averaging or exact task selection.

**Significance:** Core implementation of Multitask Prompt Tuning from https://huggingface.co/papers/2303.02861. The factorized design enables efficient multi-task learning with shared representations, while the element-wise multiplication allows task-specific modulation of the prompt space. Work from MIT-IBM Watson Research Lab.
