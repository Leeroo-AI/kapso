# Implementation: adalora/config.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/adalora/config.py`
- **Size**: 108 lines
- **Description**: Adaptive LoRA configuration with dynamic rank allocation

## Overview

AdaLoRA adapts the rank of LoRA adapters during training based on importance scores, allocating more parameters to important layers and fewer to less important ones.

## Core Configuration

### AdaLoraConfig

```python
@dataclass
class AdaLoraConfig(LoraConfig):
    # Rank configuration
    target_r: int = 8                       # Target average rank
    init_r: int = 12                        # Initial rank (must be >= target_r)

    # Training schedule
    tinit: int = 0                          # Initial warmup steps
    tfinal: int = 0                         # Final fine-tuning steps
    deltaT: int = 1                         # Rank update interval

    # Importance tracking
    beta1: float = 0.85                     # EMA for sensitivity smoothing
    beta2: float = 0.85                     # EMA for uncertainty quantification

    # Regularization
    orth_reg_weight: float = 0.5            # Orthogonal regularization coefficient

    # State
    total_step: Optional[int] = None        # Total training steps (required)
    rank_pattern: Optional[dict] = None     # Saved rank allocation
```

### Training Phases

**Phase 1: Initial Warmup** (steps 0 to tinit):
- All adapters use `init_r`
- No rank reduction
- Pre-training to gather information

**Phase 2: Budget Reduction** (steps tinit to total_step - tfinal):
- Rank budget decreases linearly
- Ranks reallocated every `deltaT` steps
- Important layers get more rank

**Phase 3: Final Fine-tuning** (steps total_step - tfinal to total_step):
- Ranks fixed
- Fine-tune reduced-rank adapters

### Validation

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.ADALORA

    # Incompatibilities
    if self.use_dora:
        raise ValueError("AdaLoRA does not support DoRA")
    if self.loftq_config:
        raise ValueError("AdaLoRA does not support LOFTQ")

    # Schedule validation
    if self.total_step is None or self.total_step <= 0:
        raise ValueError("total_step must be > 0")
    if self.tinit >= (self.total_step - self.tfinal):
        raise ValueError("No budgeting phase: decrease tinit/tfinal or increase total_step")
```

### Usage Example

```python
config = AdaLoraConfig(
    target_r=8,          # Average 8 rank
    init_r=12,           # Start with 12
    tinit=200,           # 200 steps warmup
    tfinal=1000,         # 1000 steps final tuning
    deltaT=10,           # Update ranks every 10 steps
    total_step=5000,     # Total training
    beta1=0.85,
    beta2=0.85,
    orth_reg_weight=0.5
)
```

**Timeline**:
- Steps 0-200: Warmup (r=12 everywhere)
- Steps 200-4000: Budgeting (r decreases, reallocated)
- Steps 4000-5000: Final (r fixed)

## Cross-References

- **Model**: `adalora/model.py` (rank allocation logic)
- **Layer**: `adalora/layer.py` (SVD-based adaptation)
- **Paper**: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://huggingface.co/papers/2303.10512)
