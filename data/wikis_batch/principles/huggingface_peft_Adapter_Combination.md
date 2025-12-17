# Principle: Adapter Combination

> The conceptual step of merging multiple adapters into a single composite adapter.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Model Composition |
| **Workflow** | [[huggingface_peft_Multi_Adapter_Management]] |
| **Step Number** | 4 |
| **Implementation** | [[huggingface_peft_add_weighted_adapter]] |

---

## Concept

Adapter Combination creates new behaviors by blending multiple trained adapters. This enables:

1. **Task interpolation** - Smooth transitions between specialized behaviors
2. **Multi-task fusion** - Single adapter with multiple capabilities
3. **Knowledge transfer** - Combine domain-specific knowledge
4. **Ensemble effects** - Improve robustness through combination

---

## Why This Matters

### Novel Capabilities

Combining adapters can create emergent behaviors:
```
Adapter A (Summarization) + Adapter B (Translation)
    ↓ Weighted combination
Combined Adapter (Summarize in target language)
```

### Efficient Multi-Task

| Approach | Adapters Stored | Memory at Inference |
|----------|-----------------|---------------------|
| Separate adapters | N | N × adapter_size |
| Combined adapter | 1 | 1 × adapter_size |

---

## Combination Methods

### Linear Combination

```
W_combined = w1 * W_adapter1 + w2 * W_adapter2 + ...
```

Simple weighted average of adapter weights.

### TIES Merging

Trim, Elect Sign, and Merge:
1. Trim low-magnitude changes
2. Resolve sign conflicts
3. Merge remaining values

### DARE-TIES

Combines dropout with TIES for reduced interference.

---

## Combination Patterns

### Pattern 1: Equal Blending

```python
model.add_weighted_adapter(
    adapters=["task_a", "task_b", "task_c"],
    weights=[1/3, 1/3, 1/3],
    adapter_name="equal_blend",
)
```

### Pattern 2: Primary with Auxiliaries

```python
model.add_weighted_adapter(
    adapters=["primary", "aux_1", "aux_2"],
    weights=[0.7, 0.15, 0.15],
    adapter_name="primary_enhanced",
)
```

### Pattern 3: Dynamic Interpolation

```python
def create_interpolated_adapter(model, alpha):
    model.add_weighted_adapter(
        adapters=["style_a", "style_b"],
        weights=[1-alpha, alpha],
        adapter_name=f"interp_{alpha}",
    )
```

---

## Key Decisions

### Weight Selection

| Strategy | Weights | Use Case |
|----------|---------|----------|
| Equal | [1/n, 1/n, ...] | No preference |
| Performance-based | [perf_1/sum, ...] | Metric-driven |
| Task-based | Custom | Domain knowledge |

### Combination Type Selection

| Type | Best For |
|------|----------|
| `linear` | General purpose |
| `ties` | Reducing interference |
| `dare_ties` | Large adapter sets |

---

## Relationship to Workflow

```
[Load Multiple Adapters]
       │
       ▼
[Switch Between Adapters]
       │
       ▼
[Adapter Combination] <-- This Principle
       │
       ▼
[Use Combined Adapter]
```

---

## Implementation

Adapter combination uses `add_weighted_adapter`:

[[implemented_by::Implementation:huggingface_peft_add_weighted_adapter]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:Multi_Adapter_Management]]
