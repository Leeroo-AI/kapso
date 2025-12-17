# Principle: LoRA Configuration

> The conceptual step of defining how Low-Rank Adaptation will be applied to a transformer model.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Configuration |
| **Workflow** | [[huggingface_peft_LoRA_Finetuning]] |
| **Step Number** | 2 |
| **Implementation** | [[huggingface_peft_LoraConfig]] |

---

## Concept

LoRA Configuration is the critical decision-making step where you define the architecture of your parameter-efficient fine-tuning. This step determines:

1. **Which layers receive adaptation** - Selecting target modules
2. **How much capacity the adapter has** - Setting the rank (r)
3. **How strongly the adapter influences output** - Configuring alpha scaling
4. **What training behavior to use** - Dropout, bias training, etc.

---

## Why This Matters

### Memory Efficiency

LoRA dramatically reduces trainable parameters. For a linear layer of size `d_in x d_out`:
- Full fine-tuning: `d_in * d_out` parameters
- LoRA with rank r: `(d_in + d_out) * r` parameters

Example for a 4096x4096 layer:
- Full: 16.7M parameters
- LoRA (r=16): 131K parameters (0.8% of full)

### Quality vs Efficiency Trade-off

| Rank (r) | Parameters | Typical Use Case |
|----------|------------|------------------|
| 4-8 | Minimal | Simple tasks, quick iteration |
| 16-32 | Moderate | Most fine-tuning scenarios |
| 64-128 | Higher | Complex tasks requiring capacity |
| 256+ | Significant | Approaching full fine-tune quality |

---

## Key Decisions

### 1. Target Module Selection

**Question**: Which layers should receive LoRA adapters?

**Guidelines**:
- Attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`) are most impactful
- MLP layers (`gate_proj`, `up_proj`, `down_proj`) add capacity
- `"all-linear"` is a good default for maximum flexibility

### 2. Rank Selection

**Question**: How much capacity does the adapter need?

**Guidelines**:
- Start with `r=8` or `r=16` for most tasks
- Increase if underfitting or for complex reasoning tasks
- Common ratio: `lora_alpha = 2 * r`

### 3. Scaling Strategy

**Question**: How should adapter outputs be scaled?

**Options**:
- Standard: `alpha/r` - traditional scaling
- RSLoRA: `alpha/sqrt(r)` - better stability at higher ranks

---

## Configuration Patterns

### Pattern 1: Conservative (Low Resource)

```python
# Minimal parameters, fast iteration
config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
```

### Pattern 2: Balanced (Recommended Default)

```python
# Good trade-off for most tasks
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
```

### Pattern 3: High Capacity

```python
# When task requires more adaptation
config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules="all-linear",
    use_rslora=True,  # Important at higher ranks
    task_type=TaskType.CAUSAL_LM,
)
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Rank too low | Underfitting, poor quality | Increase `r` |
| Rank too high | Overfitting, wasted compute | Reduce `r`, add dropout |
| Missing target modules | Incomplete adaptation | Use `"all-linear"` or add modules |
| Wrong task_type | Incorrect model behavior | Match to your use case |

---

## Relationship to Workflow

```
[Load Base Model]
       |
       v
[LoRA Configuration] <-- This Principle
       |
       v
[Apply PEFT to Model]
       |
       v
[Train the Adapter]
```

---

## Implementation

The configuration is instantiated using the `LoraConfig` class:

[[implemented_by::Implementation:huggingface_peft_LoraConfig]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:LoRA_Finetuning]]
