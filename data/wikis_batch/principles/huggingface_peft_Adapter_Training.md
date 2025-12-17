# Principle: Adapter Training

> The conceptual step of optimizing adapter weights through gradient descent.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Optimization |
| **Workflow** | [[huggingface_peft_LoRA_Finetuning]] |
| **Step Number** | 4 |
| **Implementation** | [[huggingface_peft_Training_Loop]] |

---

## Concept

Adapter Training is where the actual learning happens. Unlike full fine-tuning where all model parameters update, PEFT training:

1. **Keeps base weights frozen** - Original model knowledge preserved
2. **Updates only adapter weights** - Efficient gradient computation
3. **Learns task-specific representations** - Adapters encode new behavior
4. **Converges faster** - Fewer parameters mean quicker optimization

---

## Why This Matters

### Training Efficiency

| Aspect | Full Fine-tuning | PEFT Training |
|--------|-----------------|---------------|
| Parameters updated | All (~7B for 7B model) | ~0.1% (~7M) |
| Gradient memory | Full model size | Adapter size only |
| Convergence time | 10-100 epochs | 1-5 epochs |
| Overfitting risk | Higher | Lower |

### Knowledge Preservation

The frozen base model acts as a regularizer:
- Prevents catastrophic forgetting
- Maintains general capabilities
- Adapters learn task-specific adjustments

---

## Training Dynamics

### How LoRA Learns

During training, LoRA learns low-rank updates to the pretrained weights:

```
W_new = W_original + (A @ B) * scale
        ↑            ↑
        frozen       trainable
```

The adapter matrices A and B start near-zero and learn the minimal update needed for the new task.

### Loss Landscape

LoRA training operates in a constrained subspace:
- **Low-rank constraint** limits solution space
- **Initialization near zero** starts from base model behavior
- **Scaling factor** controls update magnitude

---

## Key Decisions

### 1. Learning Rate Selection

LoRA typically needs **higher learning rates** than full fine-tuning:

| Scenario | Recommended LR |
|----------|---------------|
| Standard LoRA | 1e-4 to 5e-4 |
| QLoRA (4-bit) | 2e-4 to 3e-4 |
| Full fine-tune | 1e-5 to 5e-5 |

### 2. Training Duration

| Model Size | Dataset Size | Typical Epochs |
|------------|--------------|----------------|
| 7B | 10K examples | 1-3 |
| 7B | 100K examples | 1-2 |
| 70B | 10K examples | 1-2 |

### 3. Batch Size Strategy

```python
# Effective batch size = per_device * devices * accumulation
effective_batch = 4 * 1 * 4  # = 16 effective batch size
```

---

## Monitoring Training

### Key Metrics

| Metric | What to Watch |
|--------|---------------|
| Training loss | Should decrease steadily |
| Validation loss | Watch for overfitting |
| Gradient norm | Should be stable |
| Learning rate | Follow schedule |

### Signs of Good Training

- Loss decreases smoothly
- No sudden spikes in gradients
- Validation metrics improve
- Convergence within few epochs

### Signs of Problems

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss plateaus immediately | LR too low | Increase learning rate |
| Loss explodes | LR too high | Decrease learning rate |
| Validation loss increases | Overfitting | More dropout, fewer epochs |
| No improvement | Rank too low | Increase rank |

---

## Relationship to Workflow

```
[Load Base Model]
       |
       v
[LoRA Configuration]
       |
       v
[PEFT Application]
       |
       v
[Adapter Training] <-- This Principle
       |
       v
[Adapter Saving]
```

---

## Best Practices

| Practice | Rationale |
|----------|-----------|
| Use mixed precision | 2x speedup, less memory |
| Enable gradient checkpointing | Trade compute for memory |
| Log frequently | Catch issues early |
| Validate periodically | Detect overfitting |
| Save checkpoints | Enable recovery |

---

## Implementation

Training is performed using HuggingFace Trainer or custom loops:

[[implemented_by::Implementation:huggingface_peft_Training_Loop]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:LoRA_Finetuning]]
