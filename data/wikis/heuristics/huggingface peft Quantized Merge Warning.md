# Heuristic: huggingface_peft_Quantized_Merge_Warning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Discussion|bitsandbytes issues|https://github.com/TimDettmers/bitsandbytes/issues]]
|-
! Domains
| [[domain::Quantization]], [[domain::Debugging]], [[domain::QLoRA]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
Warning: Merging LoRA adapters into 4-bit or 8-bit quantized models may produce different outputs due to rounding errors.

=== Description ===
When merging LoRA adapters into quantized base models (4-bit or 8-bit via bitsandbytes), the dequantization → merge → requantization process introduces rounding errors. This is a known limitation that can cause minor discrepancies between merged model outputs and adapter-based inference outputs.

=== Usage ===
Be aware of this heuristic when:
- Calling `model.merge_and_unload()` on a quantized model
- Calling `merge()` on 4-bit or 8-bit LoRA layers
- Exporting merged weights from a QLoRA-trained model

== The Insight (Rule of Thumb) ==

* **Warning:** Merging into quantized models will produce warnings like:
  * "Merge lora module to 4-bit linear may get different generations due to rounding errors."
  * "Merge lora module to 8-bit linear may get different generations due to rounding errors."

* **Recommendation:**
  1. **For inference:** Keep adapters separate and use `model.set_adapter()` - no rounding errors
  2. **For export:** If you must merge, verify outputs on validation set before deploying
  3. **For production:** Consider training with full-precision base model if merge quality is critical

* **Safe Merge Option:**
  * Use `safe_merge=True` to check for NaNs before completing merge
  * This catches catastrophic failures but not subtle rounding errors

* **Trade-off:** Merged models are simpler to deploy but may have slight quality degradation

== Reasoning ==

### Why Rounding Errors Occur
1. **Dequantization:** 4-bit/8-bit weights are converted to float16/float32
2. **LoRA Addition:** Delta weights are added: `W_new = W_dequantized + ΔW`
3. **Requantization:** Result is quantized back to 4-bit/8-bit

Each quantization step introduces error because not all float values can be represented exactly in reduced precision.

### Empirical Observations
In practice, the errors are usually small enough to be acceptable for most use cases. However:
- 4-bit merges have more error than 8-bit merges
- Longer sequences may accumulate more error
- Some model architectures are more sensitive than others

### Tim Dettmers' Note
The bitsandbytes author notes that 4-bit operations require defensive cloning to avoid backprop issues, suggesting the precision limitations are fundamental to the approach.

== Code Evidence ==

4-bit merge warning from `bnb.py:397-399`:
<syntaxhighlight lang="python">
warnings.warn(
    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
)
</syntaxhighlight>

8-bit merge warning from `bnb.py:110-112`:
<syntaxhighlight lang="python">
warnings.warn(
    "Merge lora module to 8-bit linear may get different generations due to rounding errors."
)
</syntaxhighlight>

Safe merge NaN check from `bnb.py:411-414`:
<syntaxhighlight lang="python">
if safe_merge and not torch.isfinite(w_data).all():
    raise ValueError(
        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
    )
</syntaxhighlight>

Defensive clone for 4-bit from `bnb.py:548-553`:
<syntaxhighlight lang="python">
# As per Tim Dettmers, for 4bit, we need to defensively clone here.
# The reason is that in some cases, an error can occur that backprop
# does not work on a manipulated view. This issue may be solved with
# newer PyTorch versions but this would need extensive testing to be
# sure.
result = result.clone()
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_merge_and_unload]]
* [[uses_heuristic::Implementation:huggingface_peft_BitsAndBytesConfig_4bit]]
* [[uses_heuristic::Workflow:huggingface_peft_QLoRA_Training]]
