# Heuristic: huggingface_peft_Quantized_Merge_Rounding

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Discussion|Tim Dettmers bitsandbytes|https://github.com/bitsandbytes-foundation/bitsandbytes]]
|-
! Domains
| [[domain::Quantization]], [[domain::Inference]], [[domain::Debugging]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Warning that merging LoRA adapters into 4-bit/8-bit quantized models may produce different generations due to rounding errors.

=== Description ===
When merging LoRA adapters into quantized base models (4-bit or 8-bit), the dequantize-merge-requantize cycle introduces rounding errors. These errors can cause the merged model to produce slightly different outputs compared to keeping the adapter separate. This is an inherent limitation of the quantization-dequantization process.

=== Usage ===
Be aware of this limitation when using `merge_and_unload()` with quantized models. For production inference where exact reproducibility matters, consider keeping adapters separate or using full-precision models for the final deployment.

== The Insight (Rule of Thumb) ==

* **Action:** Use `safe_merge=True` when calling `merge_and_unload()` to detect NaNs
* **Value:** Check merged weights for NaN/Inf before deployment
* **Trade-off:** Merged models have faster inference but may have slight output differences
* **Alternative:** Keep adapters separate for exact reproducibility

== Reasoning ==

The merge process requires:
1. **Dequantizing** the base weights from INT4/INT8 to FP16/FP32
2. **Adding** the LoRA delta (A × B × scaling)
3. **Requantizing** back to INT4/INT8

Each step introduces floating-point rounding. The requantization step is particularly lossy because it compresses the merged weights back to the lower precision format.

== Code Evidence ==

Warning from 8-bit merge in `src/peft/tuners/lora/bnb.py:110-112`:
<syntaxhighlight lang="python">
warnings.warn(
    "Merge lora module to 8-bit linear may get different generations due to rounding errors."
)
</syntaxhighlight>

Warning from 4-bit merge in `src/peft/tuners/lora/bnb.py:397-399`:
<syntaxhighlight lang="python">
warnings.warn(
    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
)
</syntaxhighlight>

Safe merge NaN check from `src/peft/tuners/lora/bnb.py:128-132`:
<syntaxhighlight lang="python">
if safe_merge and not torch.isfinite(w_data).all():
    raise ValueError(
        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
    )
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_merge_and_unload]]
* [[uses_heuristic::Workflow:huggingface_peft_Adapter_Inference]]
* [[uses_heuristic::Principle:huggingface_peft_Adapter_Merging]]

[[Category:Heuristic]]
[[Category:Quantization]]
[[Category:Debugging]]
