# Principle: Hotswap Execution

> The conceptual step of replacing adapter weights in-place without model reload.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Runtime Operation |
| **Workflow** | [[huggingface_peft_Adapter_Hotswapping]] |
| **Step Number** | 4 |
| **Implementation** | [[huggingface_peft_hotswap_adapter]] |

---

## Concept

Hotswap Execution is the actual replacement of adapter weights while maintaining:
1. **Base model state** - No reload of billions of parameters
2. **PEFT structure** - Adapter layers remain in place
3. **Compiled graph** - No recompilation if prepared properly
4. **Memory efficiency** - Weights replaced in-place

---

## Why This Matters

### Zero-Downtime Updates

```
Traditional Update:
Request → [Block] → Unload Model → Load New Model → [Resume] → Response
          ▲                                              │
          └──────────── Several seconds ─────────────────┘

Hot-Swap:
Request → [Continue serving] → Swap weights (async) → Response
                              └─ ~1ms ─┘
```

### Production Benefits

| Benefit | Traditional | Hot-Swap |
|---------|-------------|----------|
| Downtime | Seconds | ~0 |
| Memory spike | 2x model | Adapter only |
| User impact | Interrupted | None |
| Rollback speed | Seconds | ~1ms |

---

## Hot-Swap Mechanics

### What Happens During Swap

```python
# Conceptually:
for layer in peft_layers:
    # Load new weights
    new_A = load("new_adapter/lora_A")
    new_B = load("new_adapter/lora_B")

    # Replace in-place (no new allocation)
    layer.lora_A.weight.data.copy_(new_A)
    layer.lora_B.weight.data.copy_(new_B)

    # Update scaling
    layer.scaling["default"] = new_alpha / new_rank
```

### Memory Layout

```
Before:                    After:
┌─────────────────┐       ┌─────────────────┐
│   Base Model    │       │   Base Model    │  (unchanged)
│    (frozen)     │       │    (frozen)     │
├─────────────────┤       ├─────────────────┤
│  Adapter v1     │  ───► │  Adapter v2     │  (replaced in-place)
│  weights        │       │  weights        │
└─────────────────┘       └─────────────────┘
```

---

## Compatibility Requirements

### Adapters Must Match

| Property | Requirement |
|----------|-------------|
| PEFT type | Must be identical (both LoRA) |
| use_dora | Must match |
| use_rslora | Must match |
| lora_dropout | Must match |
| Target modules | New must be subset of current |

### Rank Handling

| Scenario | Behavior | Preparation Needed |
|----------|----------|-------------------|
| Same rank | Direct copy | None |
| Smaller rank | Zero-fill | None (auto-handled) |
| Larger rank | Error | `prepare_model_for_compiled_hotswap` |

---

## Execution Patterns

### Pattern 1: Scheduled Update

```python
import schedule

def daily_adapter_update():
    latest = get_latest_adapter_version()
    hotswap_adapter(model, latest, "default")
    log(f"Updated to {latest}")

schedule.every().day.at("02:00").do(daily_adapter_update)
```

### Pattern 2: Gradual Rollout

```python
def canary_deploy(model, new_adapter, rollout_pct):
    if random.random() < rollout_pct:
        hotswap_adapter(model, new_adapter, "default")
    return model
```

### Pattern 3: Feature Flags

```python
def switch_adapter_by_flag(model, flags):
    adapter = flags.get("active_adapter", "default_adapter")
    hotswap_adapter(model, adapter, "default")
```

---

## Relationship to Workflow

```
[Prepare Model for Hotswap]
       │
       ▼
[Compile Model (optional)]
       │
       ▼
[Load Initial Adapter]
       │
       ▼
[Hotswap Execution] <── This Principle
       │
       ▼
[Validate New Behavior]
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Incompatible configs | Swap fails | Match dropout, dora, rslora |
| Larger rank without prep | Shape error | Use prepare_model_for_compiled_hotswap |
| Thread safety | Race conditions | Synchronize swap with inference |

---

## Implementation

Hot-swap execution uses `hotswap_adapter`:

[[implemented_by::Implementation:huggingface_peft_hotswap_adapter]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:Adapter_Hotswapping]]
