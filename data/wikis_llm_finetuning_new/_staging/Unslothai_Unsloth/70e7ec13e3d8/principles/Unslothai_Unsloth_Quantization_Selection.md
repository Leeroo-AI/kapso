# Principle: Quantization_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|llama.cpp Quantization|https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp]]
* [[source::Blog|GGML Quantization|https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Quantization]], [[domain::Deployment]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for selecting appropriate GGUF quantization methods based on deployment requirements.

=== Description ===

Quantization selection involves choosing the right balance between model size, inference speed, and quality. GGUF supports many quantization schemes, each with different trade-offs:

**High Quality (Large Files):**
* `f32`, `f16`, `bf16`: Full or half precision, no quality loss
* `q8_0`: 8-bit, minimal quality impact

**Balanced (Recommended):**
* `q4_k_m`, `q5_k_m`: Mixed precision for optimal balance
* `q6_k`: Higher quality mixed precision

**High Compression (Quality Trade-off):**
* `q4_0`, `q3_k_m`: Standard low-bit quantization
* `q2_k`: Extreme compression

=== Usage ===

Use this principle when:
* Deciding which GGUF quantization to use
* Balancing model size against quality
* Targeting specific hardware (CPU vs GPU)
* Deploying multiple quantizations for different use cases

This step is part of GGUF export configuration.

== Theoretical Basis ==

'''K-Quants (Mixed Precision):'''
K-quant methods use different bit widths for different tensor types:

<syntaxhighlight lang="text">
q4_k_m:
  - attention.wv, feed_forward.w2: Q6_K (6-bit)
  - other tensors: Q4_K (4-bit)

q5_k_m:
  - attention.wv, feed_forward.w2: Q6_K (6-bit)
  - other tensors: Q5_K (5-bit)
</syntaxhighlight>

This improves quality by preserving precision in the most sensitive layers.

'''Selection Guidelines:'''
{| class="wikitable"
|-
! Priority !! Recommended !! Why
|-
| Quality || q8_0, bf16 || Minimal quality loss
|-
| Balance || q4_k_m, q5_k_m || Good quality/size trade-off
|-
| Size || q4_0, q3_k_m || Maximum compression
|-
| Speed || q4_k_m || Optimized kernels
|-
| Accuracy-Critical || bf16, f16 || No quantization error
|}

'''Memory Requirements (7B model):'''
{| class="wikitable"
|-
! Quantization !! File Size !! RAM Required
|-
| f16 || ~14 GB || ~16 GB
|-
| q8_0 || ~7 GB || ~8 GB
|-
| q4_k_m || ~4 GB || ~5 GB
|-
| q4_0 || ~3.5 GB || ~4 GB
|-
| q2_k || ~2.5 GB || ~3 GB
|}

== Practical Guide ==

=== Choosing Quantization ===

<syntaxhighlight lang="python">
# High quality, larger files
quantization_method = "q8_0"      # ~99.5% quality
quantization_method = "bf16"      # 100% quality

# Balanced (recommended for most uses)
quantization_method = "q4_k_m"    # ~97% quality, good size
quantization_method = "q5_k_m"    # ~98% quality

# Maximum compression
quantization_method = "q4_0"      # ~95% quality
quantization_method = "q2_k"      # ~90% quality
</syntaxhighlight>

=== Multiple Quantizations ===
<syntaxhighlight lang="python">
# Create multiple versions for different use cases
quantization_method = ["q4_k_m", "q8_0", "bf16"]

# Results in:
# - model.Q4_K_M.gguf (for CPU, limited RAM)
# - model.Q8_0.gguf (for quality-conscious deployment)
# - model.BF16.gguf (for maximum accuracy)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_ALLOWED_QUANTS]]

