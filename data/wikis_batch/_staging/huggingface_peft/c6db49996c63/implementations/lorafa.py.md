# Implementation: lorafa.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/optimizers/lorafa.py`
- **Size**: 256 lines
- **Module**: `peft.optimizers.lorafa`
- **Description**: LoRA-FA optimizer implementation for efficient LoRA training

## Overview

LoRA-FA (LoRA with Frozen-A) is a specialized optimizer that freezes the LoRA A matrices and applies a gradient transformation to B matrices to compensate. This reduces memory usage while maintaining training effectiveness. The optimizer is based on AdamW but with modified gradient computation for LoRA B matrices.

**Reference**: [LoRA-FA Paper](https://huggingface.co/papers/2308.03303)

## Core Class: LoraFAOptimizer

### Mathematical Foundation

**Standard LoRA Update**:
```
ΔW = (α/r) * B @ A
```

**LoRA-FA Gradient Transformation**:
```
g^B_FA = (r/α)² * (A^T A)^(-1) * g^B_standard
```

This minimizes: `||g_LoRA - g_FA||_F²` where `g_LoRA` is the full LoRA gradient.

### Initialization

```python
def __init__(
    self,
    params: Iterable[nn.parameter.Parameter],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    correct_bias: bool = True,
)
```

### Step Function

**Key Algorithm**:
```python
def step(self, closure: Callable = None):
    for group in self.param_groups:
        scaling_factor = group["scaling_factor"]  # α/r or α/√r
        param_list = []
        name_list = []

        for p, n in zip(group["params"], group["names"]):
            if "lora" in n:
                param_list.append(p)
                name_list.append(n)
                if len(param_list) == 2:
                    # Process A, B pair
                    A, B = param_list[0], param_list[1]
                    grad_B_orig = B.grad

                    # Compute gradient transformation
                    AA_T = A @ A.T
                    AA_T_inv = torch.linalg.pinv(AA_T + 1e-8 * I)
                    grad_B = (1 / scaling_factor²) * (grad_B_orig @ AA_T_inv)

                    # AdamW update on transformed gradient
                    exp_avg_B.mul_(beta1).add_(grad_B, alpha=(1.0 - beta1))
                    exp_avg_sq_B.mul_(beta2).addcmul_(grad_B, grad_B, value=1.0 - beta2)

                    denom_B = exp_avg_sq_B.sqrt().add_(eps)
                    B.addcdiv_(exp_avg_B, denom_B, value=-step_size)

                    param_list = []
            else:
                # Standard AdamW for non-LoRA parameters
                # ... (standard AdamW update) ...
```

### Helper Function: create_lorafa_optimizer

```python
def create_lorafa_optimizer(
    model: PeftModel,
    r: int,
    lora_alpha: int,
    lr: float,
    weight_decay: float = 0.0,
    use_rslora: bool = False
) -> Optimizer:
    """
    Creates LoRA-FA optimizer with proper configuration.

    Steps:
    1. Freeze lora_A parameters
    2. Compute scaling factor (α/r or α/√r)
    3. Create parameter groups with names
    4. Instantiate LoraFAOptimizer

    Args:
        model: PeftModel with LoRA adapters
        r: LoRA rank
        lora_alpha: LoRA scaling parameter
        lr: Learning rate
        weight_decay: L2 regularization
        use_rslora: Use RSLoRA scaling (α/√r instead of α/r)
    """
```

## Key Features

### Frozen A Matrices

**Implementation**:
```python
for name, param in model.named_parameters():
    if "lora_A" in name:
        param.requires_grad_(False)
```

**Benefits**:
- Reduces optimizer state memory by ~50%
- Faster backward pass
- Still updates B effectively

### Gradient Transformation

**Pseudo-inverse Computation**:
```python
AA_T = A @ A.T  # (r, r) matrix
AA_T_inv = torch.linalg.pinv(AA_T + delta * I)  # Regularized inverse
```

**Scaling Application**:
```python
grad_B = (1 / scaling_factor**2) * (grad_B_orig @ AA_T_inv)
```

### Mixed Precision Support

```python
device_type = infer_device()
if is_bf16_available():
    with autocast(device_type=device_type, dtype=torch.bfloat16):
        grad_B = (1 / scaling_factor**2) * (grad_B_orig @ AA_T_inv)
```

## Performance Characteristics

### Memory Savings

**Optimizer State**:
- Standard Adam: 2 × (A + B) states = 2 × (r×d + r×d)
- LoRA-FA: 2 × B states only = 2 × r×d
- **Savings**: ~50% for LoRA parameters

**Total Training Memory**:
- Gradients: 50% reduction
- Optimizer states: 50% reduction
- **Overall**: ~30-40% reduction for typical models

### Computational Overhead

**Per Update**:
- Pseudo-inverse: O(r³) for each LoRA layer
- Matrix multiplication: O(r² × d)
- Typically negligible compared to forward/backward pass

**Example** (r=64, d=4096):
- Pseudo-inverse: 64³ = 262K ops
- Matmul: 64² × 4096 = 16M ops
- Forward pass: ~100M ops
- **Overhead**: < 5%

## Usage Examples

### Basic Usage

```python
from peft import get_peft_model, LoraConfig
from peft.optimizers import create_lorafa_optimizer

# Create LoRA model
config = LoraConfig(r=64, lora_alpha=16)
model = get_peft_model(base_model, config)

# Create LoRA-FA optimizer
optimizer = create_lorafa_optimizer(
    model,
    r=64,
    lora_alpha=16,
    lr=1e-4,
    weight_decay=0.01
)

# Training loop
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### With RSLoRA

```python
optimizer = create_lorafa_optimizer(
    model,
    r=64,
    lora_alpha=16,
    lr=1e-4,
    use_rslora=True  # Uses α/√r instead of α/r
)
```

## Design Decisions

### State Management

**Per-Pair State**:
```python
state = self.state[name]  # name = "layer.lora"
state["exp_avg_B"] = torch.zeros_like(B)
state["exp_avg_sq_B"] = torch.zeros_like(B)
```

**Only B has state** (A is frozen).

### Gradient Accumulation

**Compatible with Standard Patterns**:
```python
for accumulation_step in range(num_accumulation):
    loss = model(**batch).loss / num_accumulation
    loss.backward()  # Accumulates in .grad
optimizer.step()  # Process accumulated gradients
optimizer.zero_grad()
```

### Bias Correction

**AdamW-HF Style**:
```python
if group["correct_bias"]:
    bias_correction1 = 1.0 - beta1 ** state["step"]
    bias_correction2 = 1.0 - beta2 ** state["step"]
    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
```

## Best Practices

1. **Hyperparameters**: Use same lr/weight_decay as standard AdamW
2. **Rank Selection**: Works best with r ≥ 32 (pseudo-inverse more stable)
3. **Regularization**: Add small delta (1e-8) for numerical stability
4. **Mixed Precision**: Enable for additional speedup
5. **Gradient Clipping**: Apply before optimizer.step() if needed

## Limitations

1. **LoRA-Specific**: Only works with LoRA adapters
2. **Frozen A**: A matrices cannot be updated during training
3. **Pseudo-inverse Cost**: O(r³) per layer can be expensive for very large r
4. **Initialization Dependent**: A matrices must be well-initialized

## Comparison

| Method | Memory | Compute | Flexibility |
|--------|--------|---------|-------------|
| Standard LoRA | 100% | 100% | Full |
| LoRA-FA | ~60% | ~105% | B only |
| Frozen LoRA | 0% | 0% | None |

## Cross-References

- **Paper**: [LoRA-FA: Memory-efficient Low-rank Adaptation](https://huggingface.co/papers/2308.03303)
- **Related**: `peft.tuners.lora`, `peft.optimizers.loraplus`
- **Dependencies**: `torch.optim.Optimizer`, `accelerate.utils`
