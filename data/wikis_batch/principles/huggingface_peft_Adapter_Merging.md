# Principle: Adapter Merging

> The conceptual step of permanently fusing adapter weights into the base model.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Model Transformation |
| **Workflow** | [[huggingface_peft_Adapter_Inference]] |
| **Step Number** | 5 |
| **Implementation** | [[huggingface_peft_merge_and_unload]] |

---

## Concept

Adapter Merging is an optional optimization step where the low-rank adapter weights are permanently added to the base model weights. This transforms:

**Before**: Base Model + Adapter (separate)
**After**: Single Merged Model

The result is a standard transformer model without any PEFT infrastructure.

---

## Why This Matters

### Inference Performance

| Aspect | With Adapter | Merged |
|--------|--------------|--------|
| Forward pass | Extra computation | Standard |
| Memory | Base + adapter | Base only |
| Latency | Slightly higher | Optimal |
| Compatibility | PEFT required | Any framework |

### Trade-offs

| Benefit | Cost |
|---------|------|
| Faster inference | Lose adapter flexibility |
| Simpler deployment | Cannot switch adapters |
| Framework agnostic | Larger saved model |
| No PEFT dependency | Irreversible |

---

## The Merge Operation

### Mathematical Basis

LoRA represents weight updates as low-rank matrices:
```
W_adapted = W_base + (A @ B) * scale
```

Merging computes this sum once and stores the result:
```
W_merged = W_base + (A @ B) * scale
```

### What Gets Merged

| Component | Before Merge | After Merge |
|-----------|--------------|-------------|
| Base weights | Frozen | Updated |
| LoRA A matrix | Present | Removed |
| LoRA B matrix | Present | Removed |
| Scaling factor | Applied dynamically | Baked in |

---

## When to Merge

### Recommended Scenarios

| Scenario | Reason |
|----------|--------|
| Production deployment | Maximum speed |
| Export to ONNX | Required for conversion |
| Export to TensorRT | Required for conversion |
| Single-task serving | No adapter switching needed |
| Edge deployment | Minimize complexity |

### Avoid Merging When

| Scenario | Reason |
|----------|--------|
| Multi-adapter serving | Need to switch adapters |
| A/B testing | Need different behaviors |
| Continued training | Need separate adapter weights |
| Adapter composition | Need weighted combinations |

---

## Merge Strategies

### Strategy 1: Merge and Save

```python
# Load and merge
model = PeftModel.from_pretrained(base_model, "adapter")
merged = model.merge_and_unload()

# Save as standard model
merged.save_pretrained("./merged-model")
```

### Strategy 2: Temporary Merge

```python
# Merge without removing wrapper
model.merge_adapter()

# Run optimized inference
outputs = model.generate(...)

# Unmerge if needed
model.unmerge_adapter()
```

### Strategy 3: Multi-Adapter Merge

```python
# Load multiple adapters
model.load_adapter("adapter-a", "task_a")
model.load_adapter("adapter-b", "task_b")

# Merge specific adapters
merged = model.merge_and_unload(adapter_names=["task_a", "task_b"])
```

---

## Relationship to Workflow

```
[Load Base Model]
       |
       v
[Load Trained Adapter]
       |
       v
[Configure Inference Mode]
       |
       v
[Run Inference]
       |
       v
[Adapter Merging] <-- This Principle (Optional)
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Merging during training | Lose gradient tracking | Only merge for inference |
| Expecting reversibility | merge_and_unload is permanent | Use merge_adapter() if need to unmerge |
| Wrong adapter active | Merges wrong adapter | Set active adapter first |

---

## Implementation

Adapter merging uses `merge_and_unload`:

[[implemented_by::Implementation:huggingface_peft_merge_and_unload]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:Adapter_Inference]]
