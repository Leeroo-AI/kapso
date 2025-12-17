# Implementation: save_pretrained

> API Documentation for saving trained PEFT adapter weights to disk.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/peft_model.py:L190-387` |
| **Method** | `PeftModel.save_pretrained` |
| **Paired Principle** | [[huggingface_peft_Adapter_Saving]] |
| **Parent Workflow** | [[huggingface_peft_LoRA_Finetuning]] |

---

## Purpose

`save_pretrained` saves only the trained adapter weights and configuration to disk, not the full base model. This enables sharing lightweight adapter checkpoints (typically MBs instead of GBs).

---

## API Signature

```python
model.save_pretrained(
    save_directory: str | os.PathLike,
    safe_serialization: bool = True,
    selected_adapters: list[str] | None = None,
    save_embedding_layers: str | bool = "auto",
    is_main_process: bool = True,
    convert_pissa_to_lora: str | None = None,
    path_initial_model_for_weight_conversion: str | os.PathLike | None = None,
    **kwargs,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_directory` | `str \| PathLike` | required | Directory path to save adapter files |
| `safe_serialization` | `bool` | `True` | Use safetensors format (recommended) |
| `selected_adapters` | `list[str] \| None` | `None` | Specific adapters to save; `None` saves all |
| `save_embedding_layers` | `str \| bool` | `"auto"` | Whether to save embedding layers |
| `is_main_process` | `bool` | `True` | Whether this is the main process (for distributed) |

---

## Usage Example

### Basic Save

```python
# After training
model.save_pretrained("./my-adapter")

# Saved files:
# ./my-adapter/
#   ├── adapter_config.json
#   └── adapter_model.safetensors
```

### Save Specific Adapters

```python
# With multiple adapters loaded
model.save_pretrained(
    "./adapters/task_a",
    selected_adapters=["task_a"]
)
```

### Save with PyTorch Format

```python
# Use .bin instead of .safetensors
model.save_pretrained(
    "./my-adapter",
    safe_serialization=False
)
# Creates: adapter_model.bin
```

---

## Output Files

| File | Contents |
|------|----------|
| `adapter_config.json` | PEFT configuration (rank, target_modules, etc.) |
| `adapter_model.safetensors` | Adapter weights in safetensors format |
| `adapter_model.bin` | Adapter weights in PyTorch format (if `safe_serialization=False`) |

### Example adapter_config.json

```json
{
  "base_model_name_or_path": "meta-llama/Llama-2-7b-hf",
  "bias": "none",
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "r": 16,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "task_type": "CAUSAL_LM",
  "peft_type": "LORA"
}
```

---

## Key Behaviors

### What Gets Saved

- **Adapter weights only**: LoRA A and B matrices, biases if configured
- **Configuration**: All settings needed to reconstruct the adapter
- **modules_to_save**: Any additional trainable modules specified in config

### What Does NOT Get Saved

- Base model weights (must be loaded separately)
- Optimizer state (handle separately if needed)
- Training state (epochs, steps, etc.)

### Distributed Training

For multi-GPU training, only the main process should save:

```python
if accelerator.is_main_process:
    model.save_pretrained("./my-adapter")
```

---

## File Size Comparison

| Model Size | Full Model | LoRA Adapter (r=16) | Reduction |
|------------|------------|---------------------|-----------|
| 7B params | ~14 GB | ~8-20 MB | 99.9% |
| 13B params | ~26 GB | ~15-35 MB | 99.9% |
| 70B params | ~140 GB | ~80-200 MB | 99.9% |

---

## Push to Hub

Adapters can be pushed directly to HuggingFace Hub:

```python
# Save and push
model.push_to_hub(
    "username/my-adapter",
    safe_serialization=True,
    token="hf_..."
)
```

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_PeftModel_from_pretrained]] | Load saved adapters |
| [[huggingface_peft_load_adapter]] | Load additional adapters |

---

## Source Reference

- **File**: `src/peft/peft_model.py`
- **Lines**: 190-387
- **Method**: `PeftModel.save_pretrained`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
