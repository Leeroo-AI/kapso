# Principle: Adapter Loading

> The conceptual step of loading trained adapter weights onto a base model for inference or continued training.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Initialization |
| **Workflow** | [[huggingface_peft_Adapter_Inference]], [[huggingface_peft_Multi_Adapter_Management]] |
| **Step Number** | 2 |
| **Implementation** | [[huggingface_peft_PeftModel_from_pretrained]] |

---

## Concept

Adapter Loading reverses the saving process, reconstructing a trained PEFT model from:
1. **Base model** - The original pretrained model
2. **Adapter weights** - Trained LoRA matrices from disk
3. **Configuration** - Metadata about adapter structure

This enables deployment of trained adapters without shipping the full model.

---

## Why This Matters

### Deployment Flexibility

| Scenario | Approach |
|----------|----------|
| Single task | Load base model + one adapter |
| Multi-task | Load base model + multiple adapters |
| A/B testing | Swap adapters without reload |
| Edge deployment | Ship small adapter files |

### Memory Efficiency

Loading adapters separately from the base model enables:
- **Lazy loading**: Load adapters on-demand
- **Memory sharing**: Multiple adapters share base model
- **Hot-swapping**: Replace adapters without reloading base

---

## Loading Process

### Step 1: Load Base Model

```python
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)
```

### Step 2: Load Adapter

```python
model = PeftModel.from_pretrained(
    base_model,
    "username/my-adapter",
)
```

### What Happens Internally

1. Read `adapter_config.json` to get architecture
2. Inject adapter layers into base model
3. Load `adapter_model.safetensors` weights
4. Place weights on appropriate devices

---

## Loading Patterns

### Pattern 1: Inference Mode (Default)

```python
model = PeftModel.from_pretrained(
    base_model,
    "adapter-path",
    is_trainable=False,  # Default
)
model.eval()  # Set to evaluation mode
```

### Pattern 2: Continued Training

```python
model = PeftModel.from_pretrained(
    base_model,
    "checkpoint-1000",
    is_trainable=True,  # Enable gradients
)
model.train()
```

### Pattern 3: Multi-Adapter Setup

```python
# Load first adapter
model = PeftModel.from_pretrained(base_model, "adapter-a")

# Load additional adapters
model.load_adapter("adapter-b", adapter_name="task_b")
model.load_adapter("adapter-c", adapter_name="task_c")

# Switch between adapters
model.set_adapter("task_b")
```

---

## Key Decisions

### Base Model Compatibility

The base model must be compatible with the adapter:

| Aspect | Requirement |
|--------|-------------|
| Architecture | Must match exactly |
| Hidden size | Must match exactly |
| Vocabulary | Should match (or handle mismatch) |
| Quantization | May need matching quant config |

### Device Placement

| Scenario | Recommended |
|----------|-------------|
| Single GPU | `device_map={"": 0}` |
| Multi-GPU | `device_map="auto"` |
| CPU inference | No device_map |

---

## Relationship to Workflow

```
[Load Base Model]
       |
       v
[Adapter Loading] <-- This Principle
       |
       v
[Configure Inference Mode]
       |
       v
[Run Inference]
```

---

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Size mismatch | Wrong base model | Verify base model matches training |
| Missing files | Incomplete save | Ensure both config and weights present |
| Device errors | Memory/placement | Use appropriate device_map |

---

## Implementation

Adapter loading uses `PeftModel.from_pretrained`:

[[implemented_by::Implementation:huggingface_peft_PeftModel_from_pretrained]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:Adapter_Inference]]
