# Environment: unslothai_unsloth_CUDA

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|device_type.py|unsloth/device_type.py]]
* [[source::Doc|kernels/utils.py|unsloth/kernels/utils.py]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::GPU_Computing]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

## Overview

NVIDIA/AMD/Intel GPU environment with CUDA/ROCm/XPU support, PyTorch 2.4+, Triton 3.0+, and bitsandbytes for 4-bit quantization.

### Description

This environment provides GPU-accelerated deep learning support for Unsloth model training and inference. It supports multiple GPU architectures:

- **NVIDIA CUDA**: Primary support with CUDA 11.8+ and cuDNN
- **AMD ROCm (HIP)**: Support with bitsandbytes >= 0.48.3, but with 128 blocksize (vs 64 for CUDA)
- **Intel XPU**: Support with PyTorch >= 2.6.0

The environment automatically detects GPU type and configures appropriate backends. For 4-bit quantization, bitsandbytes is required with different blocksizes for NVIDIA (64) vs AMD (128) GPUs. Pre-quantized models may not be compatible across GPU vendors due to blocksize differences.

### Usage

Use this environment for:
- **Model Loading**: `FastLanguageModel.from_pretrained()` with `load_in_4bit=True`
- **LoRA Training**: `get_peft_model()` with Triton-optimized kernels
- **Weight Merging**: `save_pretrained_merged()` for 16-bit export
- **Vision Models**: `FastVisionModel.from_pretrained()` for VLM fine-tuning

Required when:
- Any Unsloth model loading or training operation
- Using 4-bit QLoRA fine-tuning
- Using optimized Triton kernels for attention/activation

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || Windows/Mac not officially supported
|-
| Hardware || NVIDIA GPU (Ampere+), AMD GPU (gfx940+), or Intel XPU || Minimum 8GB VRAM recommended; 16GB+ for 7B models
|-
| CUDA || 11.8+ || For NVIDIA GPUs; auto-detected via `torch.cuda.is_available()`
|-
| ROCm || 5.6+ || For AMD GPUs; detected via `torch.version.hip`
|-
| XPU Driver || 2024.0+ || For Intel GPUs; requires PyTorch >= 2.6.0
|}

## Dependencies

### System Packages

* `cuda-toolkit` >= 11.8 (NVIDIA) OR `rocm` >= 5.6 (AMD) OR Intel OneAPI (XPU)
* Build tools: `gcc`, `g++`, `cmake` for Triton compilation

### Python Packages

* `torch` >= 2.4.0 (XPU requires >= 2.6.0)
* `triton` >= 3.0.0
* `transformers` >= 4.37 (for 4-bit loading support)
* `bitsandbytes` >= 0.43.3 (CUDA stream support); AMD requires >= 0.48.3
* `peft` >= 0.10.0
* `huggingface_hub`

### Version Checks in Code

```python
# From loader.py
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")
SUPPORTS_GEMMA = transformers_version >= Version("4.38")
SUPPORTS_LLAMA31 = transformers_version >= Version("4.43.2")
SUPPORTS_QWEN3 = transformers_version >= Version("4.50.3")

# From kernels/utils.py
if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd  # deprecated path

# From device_type.py
if DEVICE_TYPE == "xpu" and Version(torch.__version__) < Version("2.6.0"):
    raise RuntimeError("Intel xpu currently supports unsloth with torch.version >= 2.6.0")
```

## Credentials

The following environment variables may be required:

* `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`: HuggingFace API token for private model access
* `CUDA_VISIBLE_DEVICES`: GPU device selection (optional)

## Code Evidence

Device detection from `device_type.py:37-59`:
```python
@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    raise NotImplementedError(
        "Unsloth currently only works on NVIDIA, AMD and Intel GPUs."
    )
```

AMD compatibility from `device_type.py:81-98`:
```python
ALLOW_PREQUANTIZED_MODELS: bool = True
ALLOW_BITSANDBYTES: bool = True
if DEVICE_TYPE == "hip":
    # AMD GPUs need blocksize = 128, but our pre-quants are blocksize = 64
    if "blocksize = 64 if not HIP_ENVIRONMENT else 128" in inspect.getsource(Params4bit):
        ALLOW_PREQUANTIZED_MODELS = False
    ALLOW_BITSANDBYTES = Version(bitsandbytes.__version__) > Version("0.48.2.dev0")
```

## Related Pages

* [[requires_env::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[requires_env::Implementation:unslothai_unsloth_get_peft_model]]
* [[requires_env::Implementation:unslothai_unsloth_save_pretrained_merged]]
* [[requires_env::Implementation:unslothai_unsloth_FastVisionModel]]
