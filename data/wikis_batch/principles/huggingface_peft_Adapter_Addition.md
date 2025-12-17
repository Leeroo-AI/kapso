# Principle: Adapter Addition

> The conceptual step of loading additional adapters onto a model for multi-task capabilities.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Model Extension |
| **Workflow** | [[huggingface_peft_Multi_Adapter_Management]] |
| **Step Number** | 2 |
| **Implementation** | [[huggingface_peft_load_adapter]] |

---

## Concept

Adapter Addition extends a PeftModel with additional trained adapters. Unlike loading a new model, this:

1. **Reuses base model** - No duplicate memory for base weights
2. **Enables multi-task** - Different adapters for different tasks
3. **Supports switching** - Change behavior without reload
4. **Allows composition** - Combine adapters for novel behaviors

---

## Why This Matters

### Memory Efficiency

With a 7B base model and 3 adapters:

| Approach | Memory |
|----------|--------|
| 3 separate models | ~42 GB |
| 1 base + 3 adapters | ~14 GB + ~30 MB |
| **Savings** | ~67% |

### Deployment Flexibility

| Scenario | Solution |
|----------|----------|
| Multi-tenant serving | One adapter per tenant |
| Multi-language | One adapter per language |
| Multi-task | One adapter per task |
| A/B testing | Switch between versions |

---

## Multi-Adapter Architecture

```
                    ┌─────────────────┐
                    │   Base Model    │
                    │    (Frozen)     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌─────▼────┐        ┌────▼────┐
    │Adapter A│        │Adapter B │        │Adapter C│
    │(default)│        │ (task_b) │        │ (task_c)│
    └─────────┘        └──────────┘        └─────────┘
```

Only one adapter is active at a time (or combined via weighted adapter).

---

## Key Decisions

### 1. Adapter Naming Strategy

| Strategy | Example | Use Case |
|----------|---------|----------|
| Task-based | `"summarize"`, `"translate"` | Multi-task models |
| Version-based | `"v1"`, `"v2"` | A/B testing |
| Domain-based | `"medical"`, `"legal"` | Domain adaptation |
| Language-based | `"en"`, `"fr"`, `"de"` | Multilingual |

### 2. Compatibility Considerations

All adapters must be compatible:
- Same base model architecture
- Compatible target modules (can be subset)
- Same PEFT type (e.g., all LoRA)

### 3. Memory Planning

Plan for peak memory when multiple adapters loaded:
```
Total Memory ≈ Base Model + (Adapter Size × Number of Adapters)
```

---

## Loading Patterns

### Pattern 1: Sequential Loading

```python
# Load adapters as needed
model.load_adapter("adapter-a", "task_a")
# ... use task_a ...
model.load_adapter("adapter-b", "task_b")
# ... use task_b ...
```

### Pattern 2: Preload All

```python
# Load all adapters upfront
for name, path in adapters.items():
    model.load_adapter(path, name)

# Fast switching during inference
model.set_adapter(current_task)
```

---

## Relationship to Workflow

```
[Load Base + First Adapter]
       |
       v
[Adapter Addition] <-- This Principle
       |
       v
[Switch Active Adapter]
       |
       v
[Combine Multiple Adapters]
```

---

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Name collision | Duplicate adapter name | Use unique names |
| Incompatible adapter | Different base model | Verify compatibility |
| Memory overflow | Too many adapters | Use lazy loading |

---

## Implementation

Additional adapters are loaded using `load_adapter`:

[[implemented_by::Implementation:huggingface_peft_load_adapter]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:Multi_Adapter_Management]]
