# Environment: GPU CUDA Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|PyTorch CUDA|https://pytorch.org/get-started/locally/]]
* [[source::Doc|NVIDIA CUDA Toolkit|https://developer.nvidia.com/cuda-toolkit]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]], [[domain::GPU_Computing]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

## Overview

GPU-accelerated environment with CUDA 11.8+ or 12.x, PyTorch 2.0+, Triton 3.0+, and bitsandbytes for 4-bit QLoRA training.

### Description

This environment provides the core GPU infrastructure for Unsloth's optimized fine-tuning. It supports:
- **NVIDIA GPUs** with CUDA compute capability 7.0+ (V100, RTX 20xx and newer)
- **AMD GPUs** via ROCm/HIP backend (with some limitations on pre-quantized models)
- **Intel XPU** devices with PyTorch 2.6+ (experimental)

The environment is optimized for Ampere (A100, RTX 30xx) and Hopper (H100) architectures with bfloat16 support.

### Usage

Use this environment for all Unsloth training workflows including:
- QLoRA fine-tuning with 4-bit quantization
- Vision-language model fine-tuning
- GGUF model export (CPU compilation fallback available)

**Required for:**
- [[required_by::Implementation:unslothai_unsloth_FastLanguageModel]]
- [[required_by::Implementation:unslothai_unsloth_FastVisionModel]]
- [[required_by::Implementation:unslothai_unsloth_UnslothTrainer]]

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+), Windows 11 || Linux preferred; macOS not supported for training
|-
| Hardware || NVIDIA GPU || Minimum 16GB VRAM recommended for 7B models; 24GB+ for larger
|-
| CUDA || CUDA Toolkit 11.8+ or 12.x || CUDA 12.1+ recommended for best compatibility
|-
| Python || Python 3.9 - 3.13 || Python 3.10+ recommended
|}

### Device Detection Code Evidence

From `unsloth/device_type.py:37-59`:
```python
@functools.cache
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

## Dependencies

### System Packages

* CUDA Toolkit = 11.8 or 12.x
* cuDNN (bundled with PyTorch)
* git (for llama.cpp compilation)
* cmake (for llama.cpp compilation)

### Python Packages

Core dependencies from `pyproject.toml`:

* `torch` >= 2.0.0
* `triton` >= 3.0.0 (Linux) or `triton-windows` (Windows)
* `transformers` >= 4.51.3
* `accelerate` >= 0.34.1
* `peft` >= 0.7.1
* `trl` >= 0.18.2
* `bitsandbytes` >= 0.45.5 (for 4-bit quantization)
* `xformers` >= 0.0.22.post7 (optional, for memory-efficient attention)
* `unsloth_zoo` >= 2025.12.4

### Version Checks

From `unsloth/kernels/utils.py:41-44`:
```python
if DEVICE_TYPE == "xpu" and Version(torch.__version__) < Version("2.6.0"):
    raise RuntimeError(
        "Intel xpu currently supports unsloth with torch.version >= 2.6.0"
    )
```

From `unsloth/kernels/utils.py:62-76`:
```python
if Version(triton.__version__) >= Version("3.0.0"):
    if DEVICE_TYPE == "xpu":
        triton_tanh = tl.extra.intel.libdevice.tanh
    else:
        from triton.language.extra import libdevice
        triton_tanh = libdevice.tanh
```

## Credentials

The following environment variables may be required:

* `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`: HuggingFace API token for downloading gated models (e.g., Llama, Mistral)
* `HF_HUB_ENABLE_HF_TRANSFER`: Set to "1" for faster downloads with hf_transfer
* `CUDA_VISIBLE_DEVICES`: Control which GPUs are used for training

### Optional Environment Variables

* `UNSLOTH_ENABLE_LOGGING`: Set to "1" for verbose logging
* `UNSLOTH_DISABLE_AUTO_PADDING_FREE`: Set to "1" to disable automatic padding-free batching
* `UNSLOTH_FORCE_FLOAT32`: Set to "1" to force float32 precision
* `UNSLOTH_VLLM_STANDBY`: Set to "1" for vLLM standby mode in inference

## Related Pages

### Required By

* [[required_by::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[required_by::Implementation:unslothai_unsloth_FastVisionModel]]
* [[required_by::Implementation:unslothai_unsloth_UnslothTrainer]]
* [[required_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[required_by::Workflow:unslothai_unsloth_Vision_Model_Finetuning]]
