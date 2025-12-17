# Principle: Adapter Lifecycle

> The conceptual step of managing adapter memory through deletion and cleanup.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Resource Management |
| **Workflow** | [[huggingface_peft_Multi_Adapter_Management]] |
| **Step Number** | 6 |
| **Implementation** | [[huggingface_peft_delete_adapter]] |

---

## Concept

Adapter Lifecycle management ensures efficient use of memory resources when working with multiple adapters over time. This includes:

1. **Loading** - Adding adapters when needed
2. **Activation** - Switching between loaded adapters
3. **Deactivation** - Disabling without deletion
4. **Deletion** - Removing and freeing memory

---

## Why This Matters

### Memory Accumulation

Without lifecycle management:
```
Start: Base model (14GB)
+Load adapter A: 14GB + 50MB
+Load adapter B: 14GB + 100MB
+Load adapter C: 14GB + 150MB
... memory grows unbounded
```

### Long-Running Services

Production deployments need to:
- Load adapters on-demand
- Release unused adapters
- Prevent memory leaks

---

## Lifecycle Operations

### Complete Lifecycle

```python
# 1. Load
model.load_adapter("adapter-path", "task_a")

# 2. Use
model.set_adapter("task_a")
outputs = model.generate(inputs)

# 3. Deactivate (optional)
model.disable_adapters()

# 4. Delete
model.delete_adapter("task_a")
```

### Temporary Adapter Pattern

```python
def process_with_adapter(model, adapter_path, inputs):
    model.load_adapter(adapter_path, "temp")
    model.set_adapter("temp")
    try:
        return model.generate(inputs)
    finally:
        model.delete_adapter("temp")
```

---

## Memory Management Strategies

### Strategy 1: LRU Cache

```python
from collections import OrderedDict

class AdapterCache:
    def __init__(self, model, max_size=5):
        self.model = model
        self.cache = OrderedDict()
        self.max_size = max_size

    def get_adapter(self, name, path):
        if name in self.cache:
            self.cache.move_to_end(name)
        else:
            if len(self.cache) >= self.max_size:
                old = next(iter(self.cache))
                self.model.delete_adapter(old)
                del self.cache[old]
            self.model.load_adapter(path, name)
            self.cache[name] = path
        return name
```

### Strategy 2: Reference Counting

Track adapter usage and delete when unused.

---

## Relationship to Workflow

```
[Load Adapters]
       │
       ▼
[Switch Active Adapter]
       │
       ▼
[Combine Adapters]
       │
       ▼
[Adapter Lifecycle] <-- This Principle
       │
       ▼
[Clean Exit / Restart]
```

---

## Implementation

Adapter deletion uses `delete_adapter`:

[[implemented_by::Implementation:huggingface_peft_delete_adapter]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:Multi_Adapter_Management]]
