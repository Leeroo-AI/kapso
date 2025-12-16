# Heuristic: unslothai_unsloth_Quantization_Method_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|save.py|unsloth/save.py]]
* [[source::Blog|GGUF Quantization Guide|https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html]]
|-
! Domains
| [[domain::Optimization]], [[domain::Model_Export]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

GGUF quantization method selection guide for balancing inference speed, model size, and accuracy loss when exporting models for local deployment.

### Description

When exporting models to GGUF format for llama.cpp or Ollama, selecting the right quantization method is critical. Each method represents a different trade-off between file size, inference speed, and model quality.

Key considerations:
- **File size**: Determines download/storage requirements
- **Inference speed**: Quantized models can be faster due to memory bandwidth
- **Quality loss**: Lower bit depths incur more accuracy degradation
- **Hardware compatibility**: Some methods have specific CPU/GPU requirements

### Usage

Apply this heuristic when:
- Calling `save_pretrained_gguf()` with `quantization_method` parameter
- Deploying models to resource-constrained environments
- Optimizing for specific inference hardware
- Balancing model quality vs deployment constraints

## The Insight (Rule of Thumb)

* **Recommended Methods** (from Unsloth source):
  - **`q4_k_m`**: Best general-purpose quantization. Uses Q6_K for half of attention/feed_forward, Q4_K for rest
  - **`q5_k_m`**: Slightly higher quality than q4_k_m with ~20% larger files
  - **`q8_0`**: Minimal quality loss, but 2x larger than q4 methods
  - **`f16`/`bf16`**: No quality loss, but slow inference and large files

* **Conversion Speed**:
  - **Fast**: `not_quantized`, `fast_quantized`, `f16`, `bf16`, `q8_0`
  - **Slow**: `quantized`, `q4_k_m`, `q5_k_m` (require full model processing)

* **Trade-offs**:
  | Method | Size Factor | Quality | Speed |
  |--------|-------------|---------|-------|
  | f16 | 1.0x | 100% | Slow |
  | q8_0 | 0.5x | ~99% | Medium |
  | q5_k_m | 0.35x | ~97% | Fast |
  | q4_k_m | 0.3x | ~95% | Fastest |
  | q2_k | 0.2x | ~85% | Fastest |

* **Avoid**:
  - `f32`: Huge files, no benefit over f16
  - `q2_k` for complex tasks: Too much quality loss
  - `merged_4bit` as intermediate: Causes accuracy loss if you plan further conversions

## Reasoning

GGUF quantization works by reducing the precision of model weights from 16/32-bit floats to lower bit representations. The "k" variants (q4_k_m, q5_k_m) use importance-based quantization that preserves more precision for critical weights.

**Why q4_k_m is recommended:**
- Uses Q6_K (6-bit) for attention.wv and feed_forward.w2 tensors which are most sensitive
- Uses Q4_K (4-bit) for other tensors where precision matters less
- Achieves ~5-7x compression with minimal quality loss

**Warning about merged_4bit:**
From `save.py:268-276`:
```python
if save_method == "merged_4bit":
    raise RuntimeError(
        "Unsloth: Merging into 4bit will cause your model to lose accuracy if you plan\n"
        "to merge to GGUF or others later on. I suggest you to do this as a final step\n"
        "if you're planning to do multiple saves.\n"
        "If you are certain, change `save_method` to `merged_4bit_forced`."
    )
```

## Code Evidence

Available quantization methods from `save.py:104-131`:
```python
ALLOWED_QUANTS = {
    "not_quantized": "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized": "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized": "Recommended. Slow conversion. Fast inference, small files.",
    "f32": "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "bf16": "Bfloat16 - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "f16": "Float16 - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0": "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k": "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l": "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m": "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s": "Uses Q3_K for all tensors",
    "q4_0": "Original quant method, 4-bit.",
    "q4_1": "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s": "Uses Q4_K for all tensors",
    "q4_k": "alias for q4_k_m",
    "q5_k": "alias for q5_k_m",
    "q5_0": "Higher accuracy, higher resource usage and slower inference.",
    "q5_1": "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s": "Uses Q5_K for all tensors",
    "q6_k": "Uses Q8_K for all tensors",
    "q3_k_xs": "3-bit extra small quantization",
}
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_save_pretrained_gguf]]
* [[uses_heuristic::Workflow:unslothai_unsloth_GGUF_Export]]
