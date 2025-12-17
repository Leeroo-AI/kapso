# Principle: Hotswap Preparation

> The conceptual step of preparing a PEFT model for compiled adapter hot-swapping.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Model Preparation |
| **Workflow** | [[huggingface_peft_Adapter_Hotswapping]] |
| **Step Number** | 1 |
| **Implementation** | [[huggingface_peft_prepare_model_for_compiled_hotswap]] |

---

## Concept

Hotswap Preparation addresses a key challenge: **torch.compile creates optimized code assuming fixed tensor shapes and values**. Without preparation, swapping to an adapter with different rank or scaling would trigger recompilation.

This step:
1. **Pads weights** to maximum expected rank
2. **Converts scalings** to tensors for dynamic updates
3. **Ensures shape consistency** across all future swaps

---

## Why This Matters

### The Recompilation Problem

```
Without Preparation:
adapter_r8 → compile → adapter_r16 swap → RECOMPILE (slow!)
                                        → Shape mismatch errors

With Preparation:
adapter_r8 → prepare(target_rank=16) → compile → adapter_r16 swap → No recompile!
```

### Performance Impact

| Scenario | First Inference | Adapter Swap |
|----------|-----------------|--------------|
| No compile | ~100ms | ~1ms |
| Compile (unprepared) | ~5s | ~5s (recompile) |
| Compile (prepared) | ~5s | ~1ms |

---

## Preparation Mechanics

### Weight Padding

For a LoRA layer with rank 8, preparing for target_rank 16:

```python
# Before preparation
lora_A.shape = (4096, 8)   # Original rank
lora_B.shape = (8, 4096)

# After preparation
lora_A.shape = (4096, 16)  # Padded with zeros
lora_B.shape = (16, 4096)

# Math: (4096, 16) @ (16, 4096) with zeros in extra dims
# = Same result as (4096, 8) @ (8, 4096)
```

### Scaling Conversion

```python
# Before: float value
scaling["default"] = 2.0

# After: tensor (modifiable without recompile)
scaling["default"] = torch.tensor(2.0, device="cuda")
```

---

## Key Decisions

### 1. Determining Target Rank

Choose the maximum rank across all adapters you'll swap:

```python
adapters = [
    ("adapter-a", 8),   # rank 8
    ("adapter-b", 16),  # rank 16
    ("adapter-c", 32),  # rank 32
]
target_rank = max(r for _, r in adapters)  # 32
```

### 2. When to Prepare

| Scenario | Preparation Needed |
|----------|-------------------|
| Single adapter, compiled | Not strictly needed |
| Multiple adapters, same rank | Not strictly needed |
| Multiple adapters, different ranks | Required |
| Different alpha values | Recommended |

### 3. Memory Trade-off

Padding to larger ranks increases memory:
```
Memory increase ≈ (target_rank - actual_rank) × num_layers × 2
```

---

## Call Sequence

```
1. Load base model
       │
       ▼
2. Load first adapter
       │
       ▼
3. Prepare for hotswap <── This Principle
       │
       ▼
4. torch.compile()
       │
       ▼
5. Hot-swap adapters (no recompile)
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Prepare after compile | Error or recompilation | Always prepare before compile |
| Target rank too small | Cannot swap to larger adapter | Use max rank across all adapters |
| Forget preparation | Recompilation on each swap | Always prepare for production |

---

## Implementation

Preparation uses `prepare_model_for_compiled_hotswap`:

[[implemented_by::Implementation:huggingface_peft_prepare_model_for_compiled_hotswap]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:Adapter_Hotswapping]]
