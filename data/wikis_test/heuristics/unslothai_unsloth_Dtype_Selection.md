# Heuristic: Dtype Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|PyTorch Mixed Precision|https://pytorch.org/docs/stable/amp.html]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::Hardware]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

## Overview

Guidelines for selecting compute dtype (float16 vs bfloat16 vs float32) based on hardware capabilities and model requirements.

### Description

The choice of dtype affects training stability, memory usage, and compatibility. Unsloth automatically detects hardware support but allows manual override when needed.

### Usage

Apply this heuristic when:
- Encountering NaN/Inf loss values during training
- Working with specific model architectures (Gemma3, GPT-OSS)
- Deploying to hardware with limited dtype support

## The Insight (Rule of Thumb)

### Automatic Selection Logic

From `unsloth/models/llama.py:2207-2213`:
```python
if dtype is None:
    dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
    logger.warning_once(
        "Device does not support bfloat16. Will change to float16."
    )
    dtype = torch.float16
```

### Hardware Compatibility Matrix

| Hardware | BFloat16 Support | Recommended Dtype |
|----------|------------------|-------------------|
| NVIDIA Ampere (A100, RTX 30xx) | ✅ Yes | bfloat16 |
| NVIDIA Hopper (H100) | ✅ Yes | bfloat16 |
| NVIDIA Turing (RTX 20xx) | ❌ No | float16 |
| NVIDIA Volta (V100) | ❌ No | float16 |
| AMD MI200/MI300 | ✅ Yes | bfloat16 |
| Intel Data Center Max | ✅ Yes | bfloat16 |

### Models Requiring Float32

From `unsloth/models/loader.py:99-103`:
```python
FORCE_FLOAT32 = [
    "gemma3,",  # Add comma bc gemma3 will match gemma3n
    "gemma3n",
    "gpt_oss",
]
```

These models require float32 mixed precision due to numerical stability issues.

### When to Use Each Dtype

| Dtype | Use When | Memory | Speed | Stability |
|-------|----------|--------|-------|-----------|
| **bfloat16** | Ampere+ hardware, default choice | Low | Fast | High |
| **float16** | Older NVIDIA GPUs, consumer hardware | Low | Fast | Medium |
| **float32** | Stability issues, debugging | High | Slow | Highest |

## Reasoning

### Why BFloat16 is Preferred

BFloat16 advantages:
- Same exponent range as float32 (8 bits)
- Better numerical stability than float16
- Hardware acceleration on modern GPUs
- Handles overflow/underflow better

Float16 limitations:
- Narrow dynamic range (5-bit exponent)
- Prone to overflow with large values
- Requires loss scaling in some cases

### Model-Specific Requirements

**Gemma3/Gemma3n:**
- Uses operations that produce values exceeding float16 range
- Requires float32 for softcapping attention

From `unsloth/models/vision.py:466-478`:
```python
elif os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
    dtype = torch.float32
# ...
if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
    float32_mixed_precision = True
```

### Environment Variable Overrides

* `UNSLOTH_FORCE_FLOAT32=1`: Force float32 for debugging
* `UNSLOTH_MIXED_PRECISION`: Override mixed precision dtype
* `UNSLOTH_BFLOAT16_MIXED_PRECISION=1`: Force bfloat16 mixed precision

## Related Pages

### Used By

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[uses_heuristic::Implementation:unslothai_unsloth_FastVisionModel]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Principle:unslothai_unsloth_QLoRA_4bit_Quantization]]
