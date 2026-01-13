# Implementation: Device Type

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::GPU_Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

The `device_type.py` module provides hardware device detection and capability checking for the Unsloth library. It determines the available GPU type (NVIDIA CUDA, AMD HIP, or Intel XPU) and configures appropriate flags for quantization support based on the detected hardware.

Key responsibilities:
* Detect the available accelerator type (CUDA, HIP, or XPU)
* Determine device count for multi-GPU setups
* Configure flags for bitsandbytes and pre-quantized model support
* Handle AMD-specific warp size considerations for 4-bit quantization

== Code Reference ==

'''File:''' `unsloth/device_type.py` (127 lines)

=== Core Detection Functions ===

<syntaxhighlight lang="python">
@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))


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

=== Module Constants ===

<syntaxhighlight lang="python">
DEVICE_TYPE: str = get_device_type()
# HIP fails for autocast and other torch functions. Use CUDA instead
DEVICE_TYPE_TORCH = DEVICE_TYPE
if DEVICE_TYPE_TORCH == "hip":
    DEVICE_TYPE_TORCH = "cuda"

DEVICE_COUNT: int = get_device_count()
ALLOW_PREQUANTIZED_MODELS: bool = True
ALLOW_BITSANDBYTES: bool = True
</syntaxhighlight>

== I/O Contract ==

=== Exported Functions ===

{| class="wikitable"
|-
! Function !! Return Type !! Description
|-
| `is_hip()` || `bool` || Returns True if running on AMD HIP backend
|-
| `get_device_type()` || `str` || Returns device type string: "cuda", "hip", or "xpu"
|-
| `get_device_count()` || `int` || Returns number of available GPU devices
|}

=== Exported Constants ===

{| class="wikitable"
|-
! Constant !! Type !! Description
|-
| `DEVICE_TYPE` || `str` || Detected device type ("cuda", "hip", "xpu")
|-
| `DEVICE_TYPE_TORCH` || `str` || Device type for torch functions (HIP mapped to "cuda")
|-
| `DEVICE_COUNT` || `int` || Number of available GPU devices
|-
| `ALLOW_PREQUANTIZED_MODELS` || `bool` || Whether pre-quantized 4-bit models are supported
|-
| `ALLOW_BITSANDBYTES` || `bool` || Whether bitsandbytes quantization is supported
|}

=== AMD/HIP Warp Size Considerations ===

The module handles AMD GPU-specific block size requirements:

{| class="wikitable"
|-
! Device Type !! Warp Size !! Block Size
|-
| CUDA (NVIDIA) || 32 || 64
|-
| Radeon (Navi) || 32 || 64
|-
| Instinct (MI) || 64 || 128
|}

4-bit quantization requires block size 64, which limits support on certain AMD GPUs.

== Usage Examples ==

=== Check Device Type ===

<syntaxhighlight lang="python">
from unsloth.device_type import DEVICE_TYPE, DEVICE_COUNT, is_hip

# Get current device information
print(f"Device type: {DEVICE_TYPE}")
print(f"Device count: {DEVICE_COUNT}")
print(f"Running on AMD HIP: {is_hip()}")
</syntaxhighlight>

=== Check Quantization Support ===

<syntaxhighlight lang="python">
from unsloth.device_type import ALLOW_BITSANDBYTES, ALLOW_PREQUANTIZED_MODELS

if ALLOW_BITSANDBYTES:
    # Can use 4-bit QLoRA training
    print("4-bit QLoRA supported")
else:
    # Fall back to 16-bit or full finetuning
    print("Using 16-bit training (QLoRA not available)")

if ALLOW_PREQUANTIZED_MODELS:
    # Can load pre-quantized models from Hugging Face
    model_id = "unsloth/llama-3-8b-bnb-4bit"
</syntaxhighlight>

=== Use Device Type for Tensor Operations ===

<syntaxhighlight lang="python">
from unsloth.device_type import DEVICE_TYPE_TORCH
import torch

# Use DEVICE_TYPE_TORCH for autocast (handles HIP -> CUDA mapping)
with torch.autocast(DEVICE_TYPE_TORCH, dtype=torch.float16):
    output = model(input_ids)
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Attention_Dispatch|Attention Dispatch]] - Uses device type for backend selection
* [[Unslothai_Unsloth_Model_Registry|Model Registry]] - Registers models based on quantization support
