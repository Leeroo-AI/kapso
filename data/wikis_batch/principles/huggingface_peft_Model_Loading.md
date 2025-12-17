# Principle: Model Loading

> The conceptual step of loading a pretrained transformer model as the foundation for PEFT adaptation.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Initialization |
| **Workflow** | [[huggingface_peft_LoRA_Finetuning]] |
| **Step Number** | 1 |
| **Implementation** | [[huggingface_peft_AutoModel_from_pretrained]] |

---

## Concept

Model Loading is the foundational step where a pretrained transformer model is loaded into memory. This model serves as the "frozen base" that will receive parameter-efficient adapters. The choices made during loading significantly impact:

1. **Memory footprint** - Model precision and quantization settings
2. **Training compatibility** - Device placement and gradient handling
3. **Performance baseline** - Model architecture capabilities

---

## Why This Matters

### The Base Model is Critical

PEFT adapts a frozen base model. The quality and capabilities of that base directly determine:
- Maximum achievable task performance
- Required adapter capacity
- Hardware requirements

### Loading Configuration Impacts Training

| Loading Choice | Impact on PEFT Training |
|----------------|------------------------|
| `torch_dtype=float16` | Lower memory, faster training |
| `device_map="auto"` | Enables training large models |
| `quantization_config` | Enables QLoRA for extreme memory efficiency |

---

## Key Decisions

### 1. Model Selection

Choose a base model appropriate for your task:

| Task Type | Recommended Model Type |
|-----------|----------------------|
| Text generation | CausalLM (Llama, Mistral) |
| Classification | SequenceClassification |
| Token labeling | TokenClassification |
| Translation/Summarization | Seq2Seq |

### 2. Precision Selection

| Precision | Memory | Speed | Use Case |
|-----------|--------|-------|----------|
| `float32` | Highest | Slowest | Maximum precision needed |
| `float16` | ~50% | Fast | Default for most training |
| `bfloat16` | ~50% | Fast | Better numerical stability |
| `4-bit` | ~12.5% | Moderate | QLoRA training |

### 3. Device Strategy

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `device_map="auto"` | Automatic distribution | Multi-GPU, large models |
| `device_map={"": 0}` | Single GPU | Small models, debugging |
| CPU offload | Offload layers to CPU | Memory-constrained |

---

## Loading Patterns

### Pattern 1: Standard Full Precision

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)
```

### Pattern 2: Quantized for QLoRA

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## Relationship to Workflow

```
[Model Loading] <-- This Principle
       |
       v
[LoRA Configuration]
       |
       v
[PEFT Application]
       |
       v
[Train the Adapter]
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Wrong model type | Task mismatch | Use correct Auto class |
| OOM during loading | Model too large | Use quantization or device_map |
| Missing auth | Gated model access | Provide HF token |

---

## Implementation

Model loading uses the transformers library:

[[implemented_by::Implementation:huggingface_peft_AutoModel_from_pretrained]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:LoRA_Finetuning]]
