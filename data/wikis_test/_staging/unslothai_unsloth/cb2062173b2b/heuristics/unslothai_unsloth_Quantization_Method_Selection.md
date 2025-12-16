# Heuristic: unslothai_unsloth_Quantization_Method_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Blog|GGUF Quantization Guide|https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html]]
|-
! Domains
| [[domain::Quantization]], [[domain::Model_Export]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

Guidelines for selecting GGUF quantization method balancing file size, inference speed, and model quality.

### Description

GGUF quantization compresses model weights using various bit-widths and quantization schemes. The choice affects:
- **File size**: q2_k ~20% of f16, q8_0 ~50% of f16
- **Inference speed**: Lower bits = faster on CPU, varies on GPU
- **Quality**: More bits = closer to original model behavior

Unsloth provides three convenient presets plus direct access to all llama.cpp quantization methods.

### Usage

Use this heuristic when:
- Calling `model.save_pretrained_gguf(quantization_method=...)`
- Deploying to resource-constrained environments
- Balancing quality vs. deployment costs

## The Insight (Rule of Thumb)

### Presets (Recommended)

| Preset | Actual Method | Use Case |
|--------|---------------|----------|
| `"not_quantized"` | f16 | Maximum quality, testing |
| `"fast_quantized"` | q8_0 | Good balance, fast conversion |
| `"quantized"` | q4_k_m | Best size/quality tradeoff |

### Detailed Methods

* **`q4_k_m`** (Recommended default): Uses Q6_K for half of attention.wv and FFN.w2, Q4_K elsewhere. Best balance.
* **`q5_k_m`**: Higher quality than q4_k_m, ~25% larger files
* **`q8_0`**: Fast conversion, acceptable quality, 50% of f16 size
* **`q2_k`**: Smallest files, significant quality loss. Only for experiments.
* **`q6_k`**: Near-lossless quality, 37% of f16 size

### Decision Framework

```
Need maximum quality?       → f16 or bf16
Need fast conversion?       → q8_0 ("fast_quantized")
Need small files + quality? → q4_k_m ("quantized")
Need smallest possible?     → q2_k (quality loss)
Production deployment?      → q4_k_m or q5_k_m
```

## Reasoning

K-quant methods (q4_k_m, q5_k_m) use different quantization for different tensor types:
- **Attention weights**: Higher precision (Q6_K) preserves attention patterns
- **FFN weights**: Can tolerate more compression (Q4_K)

This mixed-precision approach maintains model behavior while achieving good compression.

From benchmarks: q4_k_m typically shows <1% perplexity increase vs f16, while q2_k shows 5-10% increase.

## Code Evidence

From `unsloth/save.py:104-131`:
```python
ALLOWED_QUANTS = {
    "not_quantized": "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized": "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized": "Recommended. Slow conversion. Fast inference, small files.",
    "f32": "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "bf16": "Bfloat16 - Fastest conversion + retains 100% accuracy.",
    "f16": "Float16  - Fastest conversion + retains 100% accuracy.",
    "q8_0": "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k": "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    # ...
}
```

Preset mapping from `unsloth/save.py:1954-1964`:
```python
if quant_method == "not_quantized":
    quant_method = "f16"
elif quant_method == "fast_quantized":
    quant_method = "q8_0"
elif quant_method == "quantized":
    quant_method = "q4_k_m"
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_save_pretrained_gguf]]
* [[uses_heuristic::Workflow:unslothai_unsloth_GGUF_Export]]
* [[uses_heuristic::Principle:unslothai_unsloth_GGUF_Conversion]]
