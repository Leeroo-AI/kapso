# Heuristic: unslothai_unsloth_AMD_GPU_Limitations

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Discussion|ROCm Issues|https://github.com/unslothai/unsloth/issues]]
|-
! Domains
| [[domain::Hardware]], [[domain::Compatibility]], [[domain::AMD]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

## Overview

Hardware compatibility heuristic: AMD GPUs (ROCm/HIP) have specific limitations with pre-quantized models and bitsandbytes versions.

### Description

AMD GPUs using ROCm (HIP backend) have different memory architecture and quantization implementations than NVIDIA CUDA. Key differences:

1. **Blocksize difference:** AMD uses blocksize=128 vs NVIDIA's blocksize=64 for 4-bit quantization
2. **Pre-quantized models:** Unsloth's pre-quantized models (e.g., `-bnb-4bit` variants) may not work correctly
3. **bitsandbytes stability:** Requires version >= 0.48.3 for stable operation

These differences are automatically detected and handled by Unsloth.

### Usage

Be aware of these limitations when:
- Running on AMD Instinct (MI250/MI300) or consumer AMD GPUs with ROCm
- Loading pre-quantized models from HuggingFace Hub
- Encountering unexpected numerical issues during training

## The Insight (Rule of Thumb)

* **Action:** Use non-pre-quantized models on AMD GPUs; let Unsloth quantize on-the-fly
* **Value:** Avoid models ending in `-bnb-4bit` or `-unsloth-bnb-4bit`
* **Trade-off:** Slightly longer model loading time (quantization happens at load time)

**Recommended approach for AMD:**
<syntaxhighlight lang="python">
# CORRECT - Use base model, let Unsloth quantize
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.1-8B-Instruct",  # Base model
    load_in_4bit = True,  # Quantize with correct AMD blocksize
)

# AVOID on AMD - Pre-quantized model may have wrong blocksize
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",  # Pre-quantized
)
</syntaxhighlight>

**If you must use pre-quantized models:**
<syntaxhighlight lang="python">
# Force exact model name to bypass auto-remapping
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    use_exact_model_name = True,  # Bypass safety checks
)
</syntaxhighlight>

## Reasoning

The 4-bit NF4 quantization in bitsandbytes groups weights into blocks. NVIDIA and AMD have different optimal block sizes:

- **NVIDIA:** blocksize = 64 (optimized for CUDA memory architecture)
- **AMD:** blocksize = 128 (required by HIP/ROCm)

When a model is pre-quantized with blocksize=64 (on NVIDIA), loading it on AMD with blocksize=128 causes:
1. Shape mismatches during dequantization
2. Numerical errors in computation
3. Potential crashes or NaN gradients

**Code evidence from device_type.py:81-98:**
<syntaxhighlight lang="python">
# Check blocksize for 4bit -> 64 for CUDA, 128 for AMD
# If AMD, we cannot load pre-quantized models for now :(
ALLOW_PREQUANTIZED_MODELS: bool = True
# HSA_STATUS_ERROR_EXCEPTION checks - sometimes AMD fails for BnB
ALLOW_BITSANDBYTES: bool = True

if DEVICE_TYPE == "hip":
    try:
        from bitsandbytes.nn.modules import Params4bit

        if "blocksize = 64 if not HIP_ENVIRONMENT else 128" in inspect.getsource(Params4bit):
            ALLOW_PREQUANTIZED_MODELS = False

        import bitsandbytes
        ALLOW_BITSANDBYTES = Version(bitsandbytes.__version__) > Version("0.48.2.dev0")
    except:
        pass
</syntaxhighlight>

**Warning message from loader.py (when AMD + 4-bit):**
<syntaxhighlight lang="python">
if not ALLOW_BITSANDBYTES and not use_exact_model_name:
    if load_in_4bit or load_in_8bit or model_name.lower().endswith("-bnb-4bit"):
        print(
            "Unsloth: AMD currently is not stable with 4bit bitsandbytes. Disabling for now."
        )
    load_in_4bit = False
</syntaxhighlight>

**AMD GPU-specific considerations:**

{| class="wikitable"
! Aspect !! NVIDIA !! AMD (ROCm)
|-
| Blocksize || 64 || 128
|-
| Pre-quantized models || Fully supported || May not work
|-
| bitsandbytes version || Any recent || >= 0.48.3
|-
| Flash Attention || Available || Available (with ROCm build)
|-
| Triton || Standard backend || Uses HIP backend
|}

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_FastLanguageModel_from_pretrained]]
* [[uses_heuristic::Principle:unslothai_unsloth_Model_Loading]]
* [[uses_heuristic::Environment:unslothai_unsloth_CUDA]]
