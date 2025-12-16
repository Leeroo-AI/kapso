# Heuristic: GGUF Quantization Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Blog|GGML Quantization Guide|https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html]]
|-
! Domains
| [[domain::Model_Deployment]], [[domain::Quantization]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

## Overview

Guide for selecting the optimal GGUF quantization method based on quality, size, and inference speed requirements.

### Description

GGUF (GPT-Generated Unified Format) supports multiple quantization levels. The choice impacts model size, inference speed, and output quality. Unsloth provides convenient access to all llama.cpp quantization methods.

### Usage

Apply this heuristic when:
- Exporting fine-tuned models to GGUF
- Deploying models to edge devices or CPUs
- Balancing model quality against resource constraints

## The Insight (Rule of Thumb)

### Quick Selection Guide

| Priority | Recommended Method | Bits per Weight | Notes |
|----------|-------------------|-----------------|-------|
| **Best Quality** | `q8_0` | 8 | Fast conversion, high quality |
| **Best Balance** | `q4_k_m` | ~4.5 | **Default choice**, best quality/size |
| **Quality Focus** | `q5_k_m` | ~5.5 | Slightly better than q4_k_m |
| **Size Focus** | `q3_k_m` | ~3.5 | Good for memory-constrained |
| **Minimal Size** | `q2_k` | ~2.5 | Significant quality loss |
| **No Quantization** | `f16` or `bf16` | 16 | Full precision, large files |

### Detailed Method Comparison

From `unsloth/save.py:104-131`:

```python
ALLOWED_QUANTS = {
    "not_quantized": "Fast conversion. Slow inference, big files.",
    "fast_quantized": "Fast conversion. OK inference, OK file size.",
    "quantized": "Slow conversion. Fast inference, small files.",
    "f32": "Not recommended. 100% accuracy but super slow.",
    "bf16": "Fastest conversion + 100% accuracy. Slow inference.",
    "f16": "Fastest conversion + 100% accuracy. Slow inference.",
    "q8_0": "Fast conversion. High resource use, acceptable quality.",
    "q4_k_m": "Recommended. Q6_K for half of attention.wv/ff.w2, else Q4_K",
    "q5_k_m": "Recommended. Q6_K for half of attention.wv/ff.w2, else Q5_K",
    "q6_k": "Uses Q8_K for all tensors",
}
```

### Unsloth Shortcuts

* `"not_quantized"` → f16 (fastest export)
* `"fast_quantized"` → q8_0 (quick but larger)
* `"quantized"` → q4_k_m (best default)

### Model Size Impact

| Method | 7B Model Size | 13B Model Size | 70B Model Size |
|--------|---------------|----------------|----------------|
| f16 | ~14GB | ~26GB | ~140GB |
| q8_0 | ~7GB | ~13GB | ~70GB |
| q4_k_m | ~4GB | ~7GB | ~40GB |
| q3_k_m | ~3GB | ~5.5GB | ~30GB |
| q2_k | ~2.5GB | ~4.5GB | ~25GB |

## Reasoning

### Why q4_k_m is Default

The K-quants (k_m, k_s, k_l) use mixed precision:
- Important tensors (attention, feed-forward) get higher precision
- Less critical tensors get lower precision
- `q4_k_m` provides the best balance for most use cases

### Quality vs. Size Trade-off

- Below q4_k_m: Noticeable quality degradation, especially for complex tasks
- Above q5_k_m: Diminishing returns on quality improvement
- q8_0: Nearly lossless but 2x the size of q4_k_m

### Hardware Considerations

* **CPU Inference:** q4_k_m or lower for reasonable speed
* **GPU Inference:** Can use q8_0 or even f16 if VRAM permits
* **Apple Silicon:** q4_k_m performs well with Metal acceleration

### Bfloat16 Hardware Check

From `unsloth/save.py:1176`:
```python
logger.warning("Unsloth: Switching bf16 to f16 due to hardware limitations")
```

Some older hardware doesn't support bfloat16, requiring fallback to float16.

## Related Pages

### Used By

* [[uses_heuristic::Implementation:unslothai_unsloth_save_to_gguf]]
* [[uses_heuristic::Workflow:unslothai_unsloth_Model_Export_GGUF]]
* [[uses_heuristic::Principle:unslothai_unsloth_GGUF_Model_Quantization]]
