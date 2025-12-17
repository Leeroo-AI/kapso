# Environment: unslothai_unsloth_CUDA

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|PyTorch|https://pytorch.org/get-started/locally/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::GPU_Acceleration]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

## Overview

GPU acceleration environment supporting NVIDIA CUDA, AMD ROCm (HIP), and Intel XPU for running Unsloth's optimized LLM training and inference kernels.

### Description

This environment defines the hardware and software stack required to run Unsloth's GPU-accelerated training operations. Unsloth supports three GPU platforms:

1. **NVIDIA CUDA** (primary): Full support with all optimizations including Triton kernels, Flash Attention, and bitsandbytes 4-bit quantization
2. **AMD ROCm (HIP)**: Partial support with some limitations on pre-quantized models
3. **Intel XPU**: Requires PyTorch >= 2.6.0

The environment auto-detects the available GPU platform and configures appropriate backends.

### Usage

This environment is **mandatory** for all Unsloth workflows. Use this environment when:
- Running any `FastLanguageModel.from_pretrained()` call
- Training with QLoRA/LoRA via `SFTTrainer` or `GRPOTrainer`
- Using Triton-based optimized kernels (RoPE, LayerNorm, Cross-Entropy)
- Saving models in merged, GGUF, or LoRA formats

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || Windows supported via WSL2; native Windows has partial support
|-
| Hardware || NVIDIA GPU (Ampere+ recommended) || Minimum 16GB VRAM for 7B models in 4-bit
|-
| Hardware (Alt) || AMD GPU (ROCm) || Limited pre-quantized model support
|-
| Hardware (Alt) || Intel XPU || Requires PyTorch >= 2.6.0
|-
| Disk || 50GB+ SSD || For model checkpoints and GGUF exports
|-
| Python || 3.9 - 3.13 || Specified in pyproject.toml
|}

## Dependencies

### System Packages

* `cuda-toolkit` >= 11.8 (for NVIDIA GPUs)
* `rocm` >= 5.0 (for AMD GPUs)
* `intel-extension-for-pytorch` (for Intel XPU)

### Python Packages

* `torch` >= 2.2.0 (>= 2.6.0 for Intel XPU)
* `triton` >= 3.0.0
* `transformers` >= 4.51.3
* `peft` >= 0.7.1
* `trl` >= 0.18.2
* `bitsandbytes` >= 0.43.3 (>= 0.48.3 for AMD ROCm)
* `accelerate` >= 0.34.1
* `xformers` (optional, for memory-efficient attention)
* `flash-attn` >= 2.6.3 (optional, for faster Gemma 2 attention)
* `unsloth_zoo` >= 2025.12.5

## Credentials

The following environment variables may be set:

* `HF_TOKEN`: HuggingFace API token for private model access
* `UNSLOTH_ENABLE_LOGGING`: Set to "1" to enable verbose logging
* `UNSLOTH_COMPILE_DEBUG`: Set to "1" for torch.compile debug mode
* `UNSLOTH_COMPILE_MAXIMUM`: Set to "1" for maximum optimization
* `UNSLOTH_DISABLE_AUTO_PADDING_FREE`: Set to "1" to disable auto padding-free

## Quick Install

<syntaxhighlight lang="bash">
# Basic installation (NVIDIA GPU with CUDA 12.1+)
pip install "unsloth[huggingface]"

# With specific CUDA version (example for CUDA 11.8 + PyTorch 2.5.0)
pip install "unsloth[cu118onlytorch250]"

# Core dependencies only
pip install torch>=2.2.0 triton>=3.0.0 transformers>=4.51.3 peft>=0.7.1 trl>=0.18.2 bitsandbytes accelerate

# For Gemma 2 with Flash Attention (optional)
pip install --no-deps "flash-attn>=2.6.3"
</syntaxhighlight>

## Code Evidence

GPU device detection from `device_type.py:37-59`:
<syntaxhighlight lang="python">
@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # Check torch.accelerator
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            raise NotImplementedError(
                "Unsloth cannot find any torch accelerator? You need a GPU."
            )
    raise NotImplementedError(
        "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."
    )
</syntaxhighlight>

Triton version check from `kernels/utils.py:62-76`:
<syntaxhighlight lang="python">
if Version(triton.__version__) >= Version("3.0.0"):
    if DEVICE_TYPE == "xpu":
        triton_tanh = tl.extra.intel.libdevice.tanh
    else:
        from triton.language.extra import libdevice
        triton_tanh = libdevice.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh
</syntaxhighlight>

Intel XPU version requirement from `kernels/utils.py:41-44`:
<syntaxhighlight lang="python">
if DEVICE_TYPE == "xpu" and Version(torch.__version__) < Version("2.6.0"):
    raise RuntimeError(
        "Intel xpu currently supports unsloth with torch.version >= 2.6.0"
    )
</syntaxhighlight>

BFloat16 support detection from `_utils.py:659-664`:
<syntaxhighlight lang="python">
if DEVICE_TYPE == "cuda":
    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version >= 8:
        SUPPORTS_BFLOAT16 = True
        # Flash Attention available for Ampere+
</syntaxhighlight>

## Common Errors

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `NotImplementedError: Unsloth currently only works on NVIDIA, AMD and Intel GPUs` || No supported GPU detected || Ensure CUDA/ROCm/XPU drivers are installed and `torch.cuda.is_available()` returns True
|-
|| `RuntimeError: Intel xpu currently supports unsloth with torch.version >= 2.6.0` || PyTorch version too old for Intel XPU || Upgrade PyTorch: `pip install torch>=2.6.0`
|-
|| `Unsloth: CUDA is not linked properly` || bitsandbytes or triton cannot find CUDA libraries || Run `sudo ldconfig /usr/lib64-nvidia` or `ldconfig /usr/local/cuda-xx.x`
|-
|| `ImportError: bitsandbytes is not installed` || Missing bitsandbytes || `pip install bitsandbytes` (4-bit QLoRA unavailable without it)
|-
|| `Unsloth: Your Flash Attention 2 installation seems to be broken` || Flash Attention CUDA version mismatch || Reinstall: `pip install --no-deps "flash-attn>=2.6.3"` or use xformers
|-
|| `xformers: Cannot use FA version` || xformers incompatible with GPU architecture || Build xformers from source for RTX 50x/Blackwell GPUs
|}

## Compatibility Notes

* **AMD GPUs (ROCm/HIP):**
  - Requires bitsandbytes >= 0.48.3 for stable 4-bit support
  - Pre-quantized models (blocksize=64) may not work due to AMD using blocksize=128
  - Set `ALLOW_PREQUANTIZED_MODELS = False` automatically on AMD

* **Intel XPU:**
  - Requires PyTorch >= 2.6.0
  - BFloat16 supported via `torch.xpu.is_bf16_supported()`

* **NVIDIA GPUs:**
  - Ampere (SM 8.0+) required for native BFloat16 and Flash Attention
  - RTX 50x/Blackwell (SM 12.0) may require xformers built from source

* **Windows:**
  - Native Windows has partial support
  - WSL2 recommended for full functionality

* **Multi-GPU:**
  - Currently limited multi-GPU support (beta available upon request)

## Related Pages

* [[requires_env::Implementation:unslothai_unsloth_import_unsloth]]
* [[requires_env::Implementation:unslothai_unsloth_FastLanguageModel_from_pretrained]]
* [[requires_env::Implementation:unslothai_unsloth_FastLanguageModel_from_pretrained_vllm]]
* [[requires_env::Implementation:unslothai_unsloth_get_peft_model]]
* [[requires_env::Implementation:unslothai_unsloth_get_chat_template]]
* [[requires_env::Implementation:unslothai_unsloth_SFTTrainer_usage]]
* [[requires_env::Implementation:unslothai_unsloth_trainer_train]]
* [[requires_env::Implementation:unslothai_unsloth_save_pretrained_merged]]
* [[requires_env::Implementation:unslothai_unsloth_GRPOTrainer_train]]
* [[requires_env::Implementation:unslothai_unsloth_save_pretrained_gguf]]
* [[requires_env::Implementation:unslothai_unsloth_push_to_hub]]
