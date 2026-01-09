# Heuristic: BFloat16_vs_Float16_Tip

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|loader.py|https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py]]
|-
! Domains
| [[domain::Training]], [[domain::Hardware]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
Use BFloat16 on Ampere+ GPUs (3000 series+) for more stable training; Float16 on older GPUs.

=== Description ===
BFloat16 (Brain Floating Point 16) and Float16 (FP16) are both 16-bit floating point formats, but with different precision characteristics:

- **BFloat16:** Same exponent range as FP32 (8 bits), less mantissa precision (7 bits)
- **Float16:** More mantissa precision (10 bits), smaller exponent range (5 bits)

BFloat16's larger exponent range prevents overflow/underflow issues common in FP16 training, making it more stable for deep learning.

=== Usage ===
Use this heuristic when:
- **Choosing dtype:** Setting `dtype` parameter in model loading
- **Debugging NaN loss:** FP16 can overflow; try BFloat16
- **New GPU purchase:** Consider Ampere+ for native BFloat16 support

== The Insight (Rule of Thumb) ==
* **Action:** Set `dtype` parameter in `FastLanguageModel.from_pretrained()` (or let Unsloth auto-detect)
* **Value:**
  - **Ampere+ GPUs (RTX 3000/4000/5000, A100, H100):** Use `dtype=torch.bfloat16` or `dtype=None` (auto-detect)
  - **Pre-Ampere GPUs (RTX 2000, V100, T4):** Use `dtype=torch.float16`
* **Trade-off:** BFloat16 more stable but slightly less precise; Float16 more precise but overflow risk
* **Auto-detection:** Unsloth automatically selects based on GPU capability

== Reasoning ==
Deep learning training involves very large and very small numbers:

1. **Gradient accumulation:** Gradients can become very large
2. **Normalization layers:** Values can span many orders of magnitude
3. **Loss scaling:** FP16 requires loss scaling to prevent underflow

BFloat16 solves these issues by maintaining FP32's dynamic range while using only 16 bits. This eliminates:
- Need for loss scaling
- Overflow during gradient accumulation
- Underflow in small gradient updates

**GPU Support:**
- **NVIDIA:** Ampere (SM80+), i.e., RTX 3000 series, A100, H100
- **AMD:** MI100+ with ROCm 5.0+
- **Intel:** XPU with PyTorch 2.6+

== Code Evidence ==

BFloat16 support detection from `_utils.py:154-165`:
<syntaxhighlight lang="python">
def is_bfloat16_supported():
    if DEVICE_TYPE == "cuda":
        major, minor = torch.cuda.get_device_capability()
        return major >= 8  # Ampere and later
    elif DEVICE_TYPE == "xpu":
        return True  # Intel XPU supports bfloat16
    elif DEVICE_TYPE == "hip":
        return True  # AMD MI series supports bfloat16
    return False
</syntaxhighlight>

Auto dtype selection in `loader.py`:
<syntaxhighlight lang="python">
if dtype is None:
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]
* [[used_by::Implementation:Unslothai_Unsloth_FastVisionModel_from_pretrained]]
