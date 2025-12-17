# Principle: PEFT Application

> The conceptual step of injecting parameter-efficient adapters into a base model.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Model Transformation |
| **Workflow** | [[huggingface_peft_LoRA_Finetuning]] |
| **Step Number** | 3 |
| **Implementation** | [[huggingface_peft_get_peft_model]] |

---

## Concept

PEFT Application is the transformation step where a frozen base model becomes a trainable adapter-augmented model. This step:

1. **Identifies target layers** - Finds modules matching the configuration
2. **Injects adapter architecture** - Wraps target layers with low-rank adapters
3. **Freezes base weights** - Ensures only adapter parameters are trainable
4. **Creates PeftModel wrapper** - Provides unified interface for training/inference

---

## Why This Matters

### Training Efficiency

By freezing the base model and only training adapter weights:
- **Memory**: Only adapter gradients stored (~0.1-1% of model size)
- **Speed**: Fewer parameters to update per step
- **Storage**: Save only adapter weights (MBs vs GBs)

### Model Architecture After PEFT Application

```
Original Linear Layer:
    input → [Linear(d_in, d_out)] → output

After LoRA Injection:
    input ─┬─→ [Frozen Linear(d_in, d_out)] ─┬─→ output
           │                                   │
           └─→ [LoRA_A(d_in, r)] ─→ [LoRA_B(r, d_out)] ─→ scale ─┘
                    (trainable)        (trainable)
```

---

## What Happens During Application

### Step 1: Module Discovery

The system iterates through all named modules in the model, matching against `target_modules`:

```python
# Pseudocode
for name, module in model.named_modules():
    if matches_target(name, config.target_modules):
        inject_adapter(module, name)
```

### Step 2: Adapter Injection

For each matched module, the original layer is replaced:

```python
# Original
model.layers[0].self_attn.q_proj = Linear(4096, 4096)

# After injection
model.layers[0].self_attn.q_proj = LoraLinear(
    base_layer=Linear(4096, 4096),  # Frozen
    lora_A=Linear(4096, r),          # Trainable
    lora_B=Linear(r, 4096),          # Trainable
    scaling=alpha/r,
)
```

### Step 3: Weight Freezing

```python
# Freeze all base model parameters
for param in model.parameters():
    param.requires_grad = False

# Enable adapter parameters
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
```

---

## Key Decisions

### When to Use get_peft_model vs inject_adapter_in_model

| Method | Returns | Use Case |
|--------|---------|----------|
| `get_peft_model()` | `PeftModel` wrapper | Standard training with HF Trainer |
| `inject_adapter_in_model()` | Mutated model directly | Custom training loops, special integrations |

### Choosing adapter_name

- Use `"default"` for single-adapter scenarios
- Use descriptive names for multi-adapter setups: `"task_a"`, `"task_b"`
- Names persist through save/load cycles

---

## Verification

After PEFT application, verify the transformation:

```python
# Check trainable parameters
model.print_trainable_parameters()
# Expected: trainable%: 0.01-1.0% for LoRA

# Verify adapter is active
print(model.active_adapter)
# Expected: "default" or your adapter_name

# List all adapters
print(model.peft_config.keys())
# Expected: dict_keys(['default'])
```

---

## Relationship to Workflow

```
[Load Base Model]
       |
       v
[LoRA Configuration]
       |
       v
[PEFT Application] <-- This Principle
       |
       v
[Train the Adapter]
```

---

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| No target modules found | ValueError during application | Check module names with `model.named_modules()` |
| All parameters trainable | `trainable%: 100%` | Config may be wrong; check `target_modules` |
| Memory not reduced | OOM during training | Ensure base model is frozen properly |

---

## Implementation

The PEFT application is performed using `get_peft_model`:

[[implemented_by::Implementation:huggingface_peft_get_peft_model]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:LoRA_Finetuning]]
