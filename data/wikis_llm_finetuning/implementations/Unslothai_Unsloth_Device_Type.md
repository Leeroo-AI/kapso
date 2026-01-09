# Implementation: Device_Type

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Device]], [[domain::Compatibility]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Device detection and capability module supporting NVIDIA CUDA, AMD HIP, and Intel XPU accelerators.

=== Description ===
This module provides device detection utilities to determine the available GPU type (CUDA, HIP, XPU) and configure Unsloth appropriately. It handles AMD Instinct vs Radeon differences, bitsandbytes compatibility, and quantization block size requirements across different GPU architectures.

=== Usage ===
Imported automatically by Unsloth to determine device capabilities. Used throughout the codebase to condition behavior on GPU type.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/device_type.py unsloth/device_type.py]
* '''Lines:''' 1-128

=== Key Functions ===
<syntaxhighlight lang="python">
def is_hip() -> bool:
    """Check if running on AMD HIP (ROCm)."""

def get_device_type() -> str:
    """
    Detect available GPU accelerator.

    Returns:
        str: One of "cuda", "hip", or "xpu"

    Raises:
        NotImplementedError: If no supported GPU found
    """

# Module-level constants
DEVICE_TYPE: str        # Detected device type
DEVICE_TYPE_TORCH: str  # Device type for torch functions (hip->cuda)
DEVICE_COUNT: int       # Number of GPUs
ALLOW_PREQUANTIZED_MODELS: bool  # Whether blocksize 64 quantization works
ALLOW_BITSANDBYTES: bool  # Whether bitsandbytes is compatible
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)
</syntaxhighlight>

== I/O Contract ==

=== Outputs (Module Constants) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| DEVICE_TYPE || str || "cuda", "hip", or "xpu"
|-
| DEVICE_TYPE_TORCH || str || Same as DEVICE_TYPE but "hip" -> "cuda" for torch compatibility
|-
| DEVICE_COUNT || int || Number of available GPUs
|-
| ALLOW_PREQUANTIZED_MODELS || bool || Whether blocksize 64 quantization is supported
|-
| ALLOW_BITSANDBYTES || bool || Whether bitsandbytes is compatible with current setup
|}

== GPU Compatibility Notes ==

{| class="wikitable"
|-
! Device !! Warp Size !! Block Size !! 4-bit QLoRA
|-
| NVIDIA CUDA || 32 || 64 || ✅ Full support
|-
| AMD Radeon (Navi) || 32 || 64 || ✅ Full support (bnb >= 0.49)
|-
| AMD Instinct (MI) || 64 || 128 || ⚠️ Limited (WIP)
|-
| Intel XPU || - || - || ✅ Requires torch >= 2.6
|}

== Usage Examples ==

=== Check Device Type ===
<syntaxhighlight lang="python">
from unsloth.device_type import DEVICE_TYPE, DEVICE_COUNT

print(f"Running on {DEVICE_TYPE} with {DEVICE_COUNT} GPU(s)")

if DEVICE_TYPE == "hip":
    print("AMD GPU detected - some features may be limited")
</syntaxhighlight>

=== Conditional Quantization ===
<syntaxhighlight lang="python">
from unsloth.device_type import ALLOW_PREQUANTIZED_MODELS

if ALLOW_PREQUANTIZED_MODELS:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",  # Pre-quantized
        load_in_4bit=True,
    )
else:
    # Fall back to on-the-fly quantization or 16-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-3-8B",
        load_in_4bit=True,  # Will quantize on load
    )
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
