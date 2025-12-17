# Principle: Adapter Switching

> The conceptual step of changing which adapter is active for model inference.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Runtime Control |
| **Workflow** | [[huggingface_peft_Multi_Adapter_Management]] |
| **Step Number** | 3 |
| **Implementation** | [[huggingface_peft_set_adapter]] |

---

## Concept

Adapter Switching is the runtime mechanism that changes which adapter's weights are used during forward passes. This enables:

1. **Zero-latency switching** - No loading or allocation
2. **Multi-task serving** - One model, many behaviors
3. **Dynamic routing** - Choose adapter per request
4. **A/B testing** - Compare adapter versions live

---

## Why This Matters

### Performance Impact

| Operation | Time | Memory |
|-----------|------|--------|
| Load new model | Seconds | Full model size |
| Load adapter | ~100ms | Adapter size |
| Switch adapter | ~0ms | None |

### Serving Architecture

```
                     Request
                        │
                        ▼
              ┌─────────────────┐
              │  Task Classifier │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    set_adapter   set_adapter   set_adapter
    ("task_a")    ("task_b")    ("task_c")
         │             │             │
         └─────────────┼─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   PeftModel     │
              │ (shared base)   │
              └─────────────────┘
```

---

## Switching Mechanics

### What Happens Internally

1. Update `active_adapter` property
2. Each adapter layer uses active adapter's weights
3. Forward pass uses updated weights
4. No memory allocation or copying

### Forward Pass with Active Adapter

```python
# Pseudocode for LoRA forward
def forward(self, x):
    base_output = self.base_layer(x)
    if self.active_adapter in self.lora_A:
        lora_output = self.lora_B[self.active_adapter](
            self.lora_A[self.active_adapter](x)
        ) * self.scaling[self.active_adapter]
        return base_output + lora_output
    return base_output
```

---

## Switching Patterns

### Pattern 1: Per-Request Routing

```python
@app.route("/generate")
def generate(request):
    task = request.headers.get("X-Task-Type", "default")
    model.set_adapter(task)
    return model.generate(request.text)
```

### Pattern 2: Batch Task Grouping

```python
# Group requests by task for efficient processing
def process_batched(requests):
    grouped = group_by_task(requests)
    results = {}
    for task, batch in grouped.items():
        model.set_adapter(task)
        results[task] = model.generate_batch(batch)
    return results
```

### Pattern 3: Fallback Chain

```python
def generate_with_fallback(model, text, tasks):
    for task in tasks:
        model.set_adapter(task)
        result = model.generate(text)
        if is_confident(result):
            return result
    return result  # Last attempt
```

---

## Key Decisions

### Thread Safety

Adapter switching is NOT thread-safe by default:
- Use separate model instances per thread, or
- Implement locking around set_adapter + generate

### Batch Processing

When processing mixed-task batches:
- Group by task first
- Switch once per group
- Avoid switching per-sample

---

## Relationship to Workflow

```
[Load Base + First Adapter]
       |
       v
[Add Additional Adapters]
       |
       v
[Adapter Switching] <-- This Principle
       |
       v
[Run Inference / Combine Adapters]
```

---

## Implementation

Adapter switching uses `set_adapter`:

[[implemented_by::Implementation:huggingface_peft_set_adapter]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:Multi_Adapter_Management]]
