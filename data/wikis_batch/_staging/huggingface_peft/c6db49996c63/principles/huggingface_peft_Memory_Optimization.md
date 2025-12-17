# Principle: Memory Optimization

> The conceptual step of preparing quantized models for efficient gradient-based training.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Model Preparation |
| **Workflow** | [[huggingface_peft_QLoRA_Training]] |
| **Step Number** | 5 |
| **Implementation** | [[huggingface_peft_prepare_model_for_kbit_training]] |

---

## Concept

Memory Optimization prepares a quantized model for training by addressing technical challenges:

1. **Gradient flow** - Ensures gradients can propagate through quantized layers
2. **Numerical stability** - Casts critical layers to full precision
3. **Memory efficiency** - Enables gradient checkpointing
4. **Training compatibility** - Sets up hooks for backward pass

---

## Why This Matters

### The Quantization Challenge

Quantized models present training challenges:
- INT4/INT8 operations don't support gradients directly
- Layer norms can become unstable in low precision
- Memory must be carefully managed for large models

### Memory Budget Breakdown

For a 7B model with QLoRA:
| Component | Memory |
|-----------|--------|
| 4-bit base weights | ~3.5 GB |
| LoRA adapters (FP16) | ~100 MB |
| Activations (with checkpointing) | ~1-2 GB |
| Optimizer states | ~200 MB |
| **Total** | ~5-6 GB |

---

## What Gets Optimized

### 1. Parameter Freezing

All base model parameters are frozen to ensure:
- Only adapter weights receive gradients
- No optimizer state for base weights
- Reduced memory footprint

### 2. Precision Casting

Critical layers are cast to higher precision:
```python
# Layer norms → float32 for stability
# Embedding layers → float32 for input gradients
```

### 3. Gradient Checkpointing

Trades compute for memory:
```
Without checkpointing: Store all activations
With checkpointing: Recompute activations during backward
Memory savings: ~40-60%
```

---

## Optimization Patterns

### Pattern 1: Standard QLoRA

```python
model = prepare_model_for_kbit_training(model)
```

### Pattern 2: Custom Checkpointing

```python
model = prepare_model_for_kbit_training(
    model,
    gradient_checkpointing_kwargs={
        "use_reentrant": False,  # Recommended for newer PyTorch
    }
)
```

### Pattern 3: Without Checkpointing

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=False,  # If memory allows
)
```

---

## Relationship to Workflow

```
[Configure Quantization]
       │
       ▼
[Load Quantized Model]
       │
       ▼
[Configure LoRA]
       │
       ▼
[Apply PEFT]
       │
       ▼
[Memory Optimization] <-- This Principle
       │
       ▼
[Train Adapter]
```

---

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM during backward | Checkpointing disabled | Enable gradient_checkpointing |
| NaN loss | Unstable layer norms | Ensure function was called |
| Slow training | Checkpointing overhead | Expected trade-off |

---

## Implementation

Memory optimization uses `prepare_model_for_kbit_training`:

[[implemented_by::Implementation:huggingface_peft_prepare_model_for_kbit_training]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:QLoRA_Training]]
