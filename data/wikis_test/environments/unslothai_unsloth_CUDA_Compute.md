# Environment: unslothai_unsloth_CUDA_Compute

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|NVIDIA CUDA|https://developer.nvidia.com/cuda-toolkit]]
* [[source::Doc|PyTorch CUDA|https://pytorch.org/get-started/locally/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::GPU_Computing]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

NVIDIA or AMD GPU environment with CUDA/ROCm support, PyTorch 2.0+, and bitsandbytes for 4-bit quantization training.

### Description

This environment provides GPU-accelerated deep learning capabilities for the Unsloth library. It supports both NVIDIA CUDA GPUs and AMD ROCm GPUs, with automatic hardware detection and configuration. The environment includes:

- **NVIDIA CUDA Support**: Full CUDA toolkit with compute capabilities for Tesla T4, A100, H100, and consumer GPUs (RTX 3090/4090)
- **AMD ROCm Support**: HIP-based GPU acceleration with adapted block sizes (128 vs 64 for CUDA)
- **Intel XPU Support**: Experimental support for Intel GPUs via torch.xpu

The device detection is handled in `unsloth/device_type.py:37-59` with checks for `torch.cuda.is_available()`, `torch.xpu.is_available()`, and HIP detection.

### Usage

Use this environment for any **model loading**, **fine-tuning**, or **inference** workflow with Unsloth. Required when:
- Loading models with `FastLanguageModel.from_pretrained()`
- Training with 4-bit quantization (QLoRA)
- Running optimized Triton kernels for attention, RoPE, and cross-entropy

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+ recommended) || Windows support via WSL2; macOS not supported for GPU
|-
| Hardware || NVIDIA GPU (Pascal or newer) OR AMD GPU || Minimum 8GB VRAM recommended; 16GB+ for 7B models
|-
| CUDA || 11.8+ or 12.x || For NVIDIA GPUs; check with `nvcc --version`
|-
| ROCm || 5.4+ || For AMD GPUs; HIP compatibility layer
|-
| Disk || 20GB+ free space || For model weights and temporary files during merging
|}

## Dependencies

### System Packages

* `cuda-toolkit` >= 11.8 (NVIDIA)
* `rocm-dev` >= 5.4 (AMD)
* `git` (for llama.cpp compilation)
* `cmake` >= 3.20 (for llama.cpp compilation)

### Python Packages

* `torch` >= 2.0.0
* `transformers` >= 4.37.0 (for 4-bit loading support)
* `bitsandbytes` >= 0.43.0 (for 4-bit quantization)
* `triton` >= 3.0.0 (for optimized kernels)
* `peft` >= 0.7.0 (for LoRA adapters)
* `accelerate` >= 0.20.0
* `xformers` (optional, for flash attention)
* `flash-attn` >= 2.6.3 (optional, for Gemma 2 softcapping)

## Credentials

The following environment variables may be needed:

* `HF_TOKEN`: HuggingFace API token for downloading gated models
* `CUDA_VISIBLE_DEVICES`: GPU selection (e.g., "0,1" for multi-GPU)

## Code Evidence

From `unsloth/device_type.py:37-59`:
```python
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # ...
    raise NotImplementedError(
        "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."
    )
```

From `unsloth/device_type.py:81-98` (AMD-specific checks):
```python
ALLOW_PREQUANTIZED_MODELS: bool = True
ALLOW_BITSANDBYTES: bool = True
if DEVICE_TYPE == "hip":
    # AMD GPUs need blocksize = 128, but pre-quants are blocksize = 64
    if "blocksize = 64 if not HIP_ENVIRONMENT else 128" in inspect.getsource(Params4bit):
        ALLOW_PREQUANTIZED_MODELS = False
    ALLOW_BITSANDBYTES = Version(bitsandbytes.__version__) > Version("0.48.2.dev0")
```

## Related Pages

* [[requires_env::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[requires_env::Implementation:unslothai_unsloth_get_peft_model]]
