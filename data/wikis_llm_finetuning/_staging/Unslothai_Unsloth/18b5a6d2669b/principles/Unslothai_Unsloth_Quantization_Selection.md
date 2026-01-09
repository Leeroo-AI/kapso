# Principle: Quantization_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|llama.cpp Quantization|https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md]]
* [[source::Paper|GGML Format|https://github.com/ggerganov/ggml]]
|-
! Domains
| [[domain::Quantization]], [[domain::GGUF]], [[domain::Model_Compression]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Selection of appropriate quantization methods for GGUF export based on accuracy/size/speed trade-offs.

=== Description ===

Quantization Selection determines the bit-width and quantization scheme for GGUF files:

* **Higher precision** (f16, bf16, q8_0): Larger files, faster conversion, higher accuracy
* **Lower precision** (q4_k_m, q5_k_m): Smaller files, slower conversion, acceptable accuracy
* **Ultra-low precision** (q2_k, q3_k_m): Smallest files, significant accuracy loss

=== Usage ===

Choose quantization based on deployment constraints:
* Local deployment with limited VRAM: q4_k_m, q5_k_m
* Cloud deployment with more resources: q8_0, f16
* Accuracy-critical applications: bf16, f16

== Theoretical Basis ==

=== Quantization Methods ===

| Method | Bits per Weight | Use Case |
|--------|-----------------|----------|
| f32 | 32 | Debugging only (very slow) |
| f16/bf16 | 16 | Full precision, slow inference |
| q8_0 | 8 | High quality, balanced |
| q6_k | 6 | Good quality, smaller |
| q5_k_m | 5 | Recommended balance |
| q4_k_m | 4 | Recommended small |
| q3_k_m | 3 | Aggressive compression |
| q2_k | 2 | Maximum compression |

=== K-Quant Variants ===

* **_s (small)**: Uses same quant for all tensors
* **_m (medium)**: Higher precision for attention/FFN key tensors
* **_l (large)**: Even higher precision for critical tensors

=== Quality vs Size Trade-off ===

<math>
\text{Perplexity Increase} \approx \frac{k}{bits^2}
</math>

Where k depends on model architecture and training data.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_ALLOWED_QUANTS]]

