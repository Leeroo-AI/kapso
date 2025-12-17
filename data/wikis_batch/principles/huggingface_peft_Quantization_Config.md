# Principle: Quantization Configuration

> The conceptual step of configuring model quantization for memory-efficient QLoRA training.

---

## Overview

| Property | Value |
|----------|-------|
| **Principle Type** | Configuration |
| **Workflow** | [[huggingface_peft_QLoRA_Training]] |
| **Step Number** | 1 |
| **Implementation** | [[huggingface_peft_BitsAndBytesConfig]] |

---

## Concept

Quantization Configuration defines how a large language model will be compressed into a lower-precision format. For QLoRA, 4-bit quantization enables:

1. **Dramatic memory reduction** - ~4x smaller than FP16
2. **Training large models** - 70B models on single consumer GPUs
3. **Minimal quality loss** - NF4 quantization preserves model quality

---

## Why This Matters

### Memory Requirements

| Model Size | FP16 | 4-bit QLoRA |
|------------|------|-------------|
| 7B | ~14 GB | ~4 GB |
| 13B | ~26 GB | ~8 GB |
| 70B | ~140 GB | ~35 GB |

### Enabling Factors

QLoRA works because:
- LoRA adapters remain in full precision (trainable)
- Base model is frozen and can be quantized
- Gradients flow through dequantized activations

---

## Key Decisions

### 1. Quantization Type

| Type | Description | Recommendation |
|------|-------------|----------------|
| `nf4` | Normal Float 4-bit | Recommended for LLMs |
| `fp4` | Floating Point 4-bit | Alternative option |

### 2. Compute Dtype

| Dtype | Speed | Quality |
|-------|-------|---------|
| `float16` | Fast | Good |
| `bfloat16` | Fast | Better stability |
| `float32` | Slower | Maximum precision |

### 3. Double Quantization

Enabling `bnb_4bit_use_double_quant` further reduces memory by quantizing the quantization constants.

---

## Configuration Patterns

### Pattern 1: Memory-Optimized

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Extra savings
)
```

### Pattern 2: Quality-Focused

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Better numerics
    bnb_4bit_use_double_quant=False,
)
```

---

## Relationship to Workflow

```
[Quantization Config] <-- This Principle
       │
       ▼
[Load Quantized Model]
       │
       ▼
[Configure LoRA for QLoRA]
       │
       ▼
[Apply PEFT]
```

---

## Implementation

Quantization is configured using `BitsAndBytesConfig`:

[[implemented_by::Implementation:huggingface_peft_BitsAndBytesConfig]]

---

[[Category:Principle]]
[[Category:huggingface_peft]]
[[Category:QLoRA_Training]]
