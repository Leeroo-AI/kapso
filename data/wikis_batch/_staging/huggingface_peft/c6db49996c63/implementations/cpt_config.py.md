# Implementation: cpt/config.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/cpt/config.py`
- **Size**: 99 lines
- **Description**: Context-aware Prompt Tuning (CPT) configuration

## Overview

CPT extends prompt tuning with context-aware features including token-type masking, weighted loss, and projection-based optimization. It's designed specifically for causal language modeling tasks.

**Reference**: [CPT Paper](https://huggingface.co/papers/2410.17222)

## Core Configuration

### CPTConfig

```python
@dataclass
class CPTConfig(PromptLearningConfig):
    # Token configuration
    cpt_token_ids: Optional[list[int]] = None           # Prompt token IDs
    cpt_mask: Optional[list[int]] = None                # Token mask
    cpt_tokens_type_mask: Optional[list[int]] = None    # Token type indicator

    # Loss configuration
    opt_weighted_loss_type: Optional[Literal["none", "decay"]] = "none"
    opt_loss_decay_factor: Optional[float] = 1.0

    # Projection configuration
    opt_projection_epsilon: Optional[float] = 0.1
    opt_projection_format_epsilon: Optional[float] = 0.1

    # Tokenizer
    tokenizer_name_or_path: Optional[str] = None
```

### Key Features

**Token-Type Masking**: Different treatment for input/format/output tokens
**Weighted Loss**: Optional exponential decay for loss weighting
**Projection-Based**: Epsilon-constrained projection for stability

## Validation

```python
def __post_init__(self):
    # Must be CAUSAL_LM
    if self.task_type != TaskType.CAUSAL_LM:
        raise ValueError(f"CPT only supports CAUSAL_LM")

    # Initialize defaults
    if self.cpt_token_ids is None:
        self.cpt_token_ids = [0]

    # Validate lengths
    if not (len(self.cpt_token_ids) == len(self.cpt_mask) == len(self.cpt_tokens_type_mask)):
        raise ValueError("All token lists must have same length")
```

## Cross-References

- **Paper**: [Context-aware Prompt Tuning](https://huggingface.co/papers/2410.17222)
- **Related**: `peft.tuners.prompt_tuning`
