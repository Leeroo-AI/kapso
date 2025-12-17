# Implementation: prepare_model_for_compiled_hotswap

> API Documentation for preparing PEFT models for hot-swapping with torch.compile.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/utils/hotswap.py:L268-367` |
| **Function** | `prepare_model_for_compiled_hotswap` |
| **Paired Principle** | [[huggingface_peft_Hotswap_Preparation]] |
| **Parent Workflow** | [[huggingface_peft_Adapter_Hotswapping]] |

---

## Purpose

`prepare_model_for_compiled_hotswap` prepares a PEFT model for hot-swapping adapters while maintaining compatibility with `torch.compile`. This is necessary when:
- Adapters have different ranks
- Adapters have different alpha values
- You want to avoid recompilation when swapping

---

## API Signature

```python
from peft.utils.hotswap import prepare_model_for_compiled_hotswap

prepare_model_for_compiled_hotswap(
    model: torch.nn.Module,
    *,
    target_rank: int | None = None,
    config: LoraConfig | dict[str, LoraConfig] | None = None,
    check_compiled: Literal["error", "warn", "ignore"] = "error",
) -> None
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | required | PeftModel with first adapter loaded |
| `target_rank` | `int \| None` | `None` | Maximum rank to pad adapters to |
| `config` | `LoraConfig \| dict` | `None` | Config(s) to update with target rank |
| `check_compiled` | `str` | `"error"` | How to handle already-compiled models |

---

## Usage Example

### Standard Usage

```python
from peft import PeftModel
from peft.utils.hotswap import prepare_model_for_compiled_hotswap, hotswap_adapter

# Load first adapter
model = PeftModel.from_pretrained(base_model, "adapter-r16")

# Prepare for hotswap (before compile!)
prepare_model_for_compiled_hotswap(
    model,
    target_rank=32,  # Max rank across all adapters
)

# Now compile
model = torch.compile(model)

# Hot-swap to different adapter
hotswap_adapter(model, "adapter-r8", "default", torch_device="cuda")
```

---

## Key Behaviors

### What Gets Prepared

1. **Scaling tensors**: Converted to `torch.Tensor` for dynamic updates
2. **Weight padding**: LoRA matrices padded to `target_rank` if specified
3. **Config update**: Rank patterns updated if config provided

### Padding Strategy

For adapters with rank < target_rank:
```
LoRA_A: (d_in, r) → (d_in, target_rank) with zeros
LoRA_B: (r, d_out) → (target_rank, d_out) with zeros
```

This maintains mathematical equivalence while ensuring consistent tensor shapes.

### Call Order

**Critical**: Must be called AFTER loading first adapter, BEFORE `torch.compile`:

```python
model = PeftModel.from_pretrained(...)  # 1. Load adapter
prepare_model_for_compiled_hotswap(...)  # 2. Prepare
model = torch.compile(model)             # 3. Compile
hotswap_adapter(...)                     # 4. Swap (no recompile!)
```

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Call before compiling` | Model already compiled | Call prepare before torch.compile |
| `ValueError: No adapter layers found` | No adapter loaded | Load adapter first |

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_hotswap_adapter]] | Perform the actual swap |
| `torch.compile` | Compile model after preparation |

---

## Source Reference

- **File**: `src/peft/utils/hotswap.py`
- **Lines**: 268-367
- **Function**: `prepare_model_for_compiled_hotswap`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
