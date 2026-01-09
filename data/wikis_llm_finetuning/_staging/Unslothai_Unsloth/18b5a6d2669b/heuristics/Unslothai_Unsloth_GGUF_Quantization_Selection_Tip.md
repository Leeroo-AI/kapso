# Heuristic: GGUF_Quantization_Selection_Tip

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|save.py|https://github.com/unslothai/unsloth/blob/main/unsloth/save.py]]
|-
! Domains
| [[domain::Quantization]], [[domain::Deployment]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
Use q4_k_m for balanced quality/size, q8_0 for maximum quality, or q2_k for maximum compression.

=== Description ===
GGUF quantization reduces model file size for local deployment. The quantization method determines the trade-off between model quality (perplexity) and file size/inference speed.

Unsloth supports all standard llama.cpp quantization methods through the `quantization_method` parameter in `save_pretrained_gguf()`.

=== Usage ===
Use this heuristic when:
- **Deploying to consumer hardware:** Need to fit model in limited VRAM/RAM
- **Optimizing inference speed:** Lower precision = faster inference on CPU
- **Balancing quality vs size:** Choosing between perplexity loss and model size

== The Insight (Rule of Thumb) ==
* **Action:** Set `quantization_method` in `save_pretrained_gguf()` or `push_to_hub_gguf()`
* **Values:**

{| class="wikitable"
|-
! Method !! Bits/Weight !! Use Case !! Quality Loss
|-
| `f16` || 16 || Maximum quality, baseline || None
|-
| `bf16` || 16 || Maximum quality (if supported) || None
|-
| `q8_0` || 8 || High quality, moderate size || Minimal
|-
| `q6_k` || 6.5 || Good quality, smaller size || Very low
|-
| `q5_k_m` || 5.5 || Balanced quality/size || Low
|-
| `q4_k_m` || 4.5 || **Recommended default** || Acceptable
|-
| `q3_k_m` || 3.4 || Small models, some quality loss || Moderate
|-
| `q2_k` || 2.6 || Maximum compression || Noticeable
|}

* **Trade-off:** Lower bits = smaller file = faster inference = lower quality
* **Default:** `q4_k_m` ("quantized" preset)

== Reasoning ==
Quantization works by representing weights with fewer bits:

1. **k-quant methods** (q4_k_m, q5_k_m, etc.) use mixed precision where important weights get more bits
2. **m suffix** indicates medium quality variant (vs s=small, l=large)
3. **8-bit (q8_0)** provides near-lossless quality at 50% size reduction
4. **4-bit (q4_k_m)** provides 75% size reduction with acceptable quality loss

**Recommendation Matrix:**

| Scenario | Method |
|----------|--------|
| Quality-critical applications | q8_0 or f16 |
| General purpose chat | q4_k_m |
| Resource-constrained deployment | q3_k_m or q2_k |
| Ollama/LM Studio default | q4_k_m |

== Code Evidence ==

Quantization options from `save.py:104-131`:
<syntaxhighlight lang="python">
ALLOWED_QUANTS = {
    "not_quantized": "No quantization (float16)",
    "fast_quantized": "Q8_0 quantization (8-bit)",
    "quantized": "Q4_K_M quantization (4-bit)",
    "q4_k_m": "Q4_K_M quantization",
    "q5_k_m": "Q5_K_M quantization",
    "q8_0": "Q8_0 quantization",
    "q2_k": "Q2_K quantization (2-bit)",
    "q3_k_m": "Q3_K_M quantization",
    "q6_k": "Q6_K quantization",
    "f16": "Float16 (no quantization)",
    "bf16": "BFloat16 (no quantization)",
}
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Unslothai_Unsloth_save_to_gguf]]
* [[used_by::Implementation:Unslothai_Unsloth_push_to_hub_gguf]]
* [[used_by::Implementation:Unslothai_Unsloth_ALLOWED_QUANTS]]
