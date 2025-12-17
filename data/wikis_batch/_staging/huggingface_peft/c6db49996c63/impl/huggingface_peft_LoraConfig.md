# Implementation: LoraConfig

> API Documentation for configuring Low-Rank Adaptation (LoRA) adapters in the PEFT library.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/tuners/lora/config.py:L322-880` |
| **Class/Function** | `LoraConfig` |
| **Paired Principle** | [[huggingface_peft_LoRA_Configuration]] |
| **Parent Workflow** | [[huggingface_peft_LoRA_Finetuning]] |

---

## Purpose

`LoraConfig` is a dataclass configuration object that stores all parameters needed to create and configure a LoRA adapter. It determines which layers receive LoRA adaptation, the rank of the low-rank matrices, scaling factors, dropout rates, and initialization strategies.

---

## API Signature

```python
from peft import LoraConfig

config = LoraConfig(
    r: int = 8,
    target_modules: Optional[Union[List[str], str]] = None,
    exclude_modules: Optional[Union[List[str], str]] = None,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    fan_in_fan_out: bool = False,
    bias: Literal["none", "all", "lora_only"] = "none",
    use_rslora: bool = False,
    modules_to_save: Optional[List[str]] = None,
    init_lora_weights: bool | Literal["gaussian", "eva", "olora", "pissa", "corda", "loftq", "orthogonal"] = True,
    layers_to_transform: Optional[Union[List[int], int]] = None,
    layers_pattern: Optional[Union[List[str], str]] = None,
    rank_pattern: Optional[dict] = {},
    alpha_pattern: Optional[dict] = {},
    use_dora: bool = False,
    task_type: TaskType = None,
)
```

---

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | `int` | `8` | LoRA attention dimension (the "rank"). Higher values = more capacity but more parameters |
| `lora_alpha` | `int` | `8` | Alpha parameter for LoRA scaling. Effective scaling is `alpha/r` (or `alpha/sqrt(r)` with rslora) |
| `target_modules` | `List[str] \| str \| None` | `None` | Module names to apply LoRA. Regex supported. Use `"all-linear"` to target all linear layers |
| `exclude_modules` | `List[str] \| str \| None` | `None` | Module names to exclude from LoRA adaptation |
| `lora_dropout` | `float` | `0.0` | Dropout probability applied to LoRA layers |
| `task_type` | `TaskType` | `None` | The task type (e.g., `CAUSAL_LM`, `SEQ_CLS`, `TOKEN_CLS`) |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bias` | `Literal["none", "all", "lora_only"]` | `"none"` | Which biases to train alongside LoRA weights |
| `use_rslora` | `bool` | `False` | Use Rank-Stabilized LoRA scaling (`alpha/sqrt(r)`) |
| `use_dora` | `bool` | `False` | Enable DoRA (Weight-Decomposed Low-Rank Adaptation) |
| `init_lora_weights` | `bool \| str` | `True` | Weight initialization strategy |
| `modules_to_save` | `List[str]` | `None` | Additional modules to train and save (e.g., classifier heads) |

### Layer Selection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layers_to_transform` | `List[int] \| int` | `None` | Specific layer indices to apply LoRA |
| `layers_pattern` | `List[str] \| str` | `None` | Pattern to identify layer module list (e.g., `"layers"`, `"h"`) |
| `rank_pattern` | `dict` | `{}` | Per-layer rank overrides via regex patterns |
| `alpha_pattern` | `dict` | `{}` | Per-layer alpha overrides via regex patterns |

---

## Usage Example

### Basic Configuration

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### All-Linear Configuration

```python
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",  # Targets all linear layers except output
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)
```

### Advanced Configuration with DoRA

```python
config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,  # Weight-Decomposed LoRA
    use_rslora=True,  # Rank-Stabilized scaling
    task_type=TaskType.CAUSAL_LM,
)
```

---

## Return Value

Returns a `LoraConfig` instance that can be passed to `get_peft_model()` or `PeftModel.from_pretrained()`.

---

## Key Behaviors

### Target Module Resolution

1. If `target_modules=None`: Uses model architecture defaults
2. If `target_modules="all-linear"`: Targets all `nn.Linear` and `Conv1D` layers (excluding output layer)
3. If `target_modules` is a list: Exact match or suffix match on module names
4. If `target_modules` is a string: Regex full match against module names

### Scaling Calculation

- Standard: `scaling = lora_alpha / r`
- With RSLoRA: `scaling = lora_alpha / sqrt(r)`

### Initialization Strategies

| Strategy | Description |
|----------|-------------|
| `True` | Microsoft default: LoRA A kaiming, LoRA B zeros |
| `False` | Random initialization (for debugging) |
| `"gaussian"` | Gaussian scaled by rank |
| `"eva"` | Data-driven Explained Variance Adaptation |
| `"pissa"` | Principal Singular Values and Vectors Adaptation |
| `"loftq"` | LoftQ quantization-aware initialization |

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_get_peft_model]] | Uses config to create PeftModel |
| [[huggingface_peft_PeftModel_from_pretrained]] | Loads adapter with associated config |

---

## Source Reference

- **File**: `src/peft/tuners/lora/config.py`
- **Lines**: 322-880
- **Class**: `LoraConfig(PeftConfig)`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
