# Implementation: Grouped_GEMM_Tuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::MoE]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Kernel configuration dataclasses for manually tuned grouped GEMM parameters.

=== Description ===
This module defines dataclasses for kernel configuration parameters used when manually tuning grouped GEMM kernels (without autotuning). It includes separate configs for forward, dW backward, and dX backward kernels with fields for block sizes, warp counts, and pipeline stages.

=== Usage ===
Used when passing manual kernel configurations to grouped_gemm_forward instead of using autotune=True.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/kernels/tuning.py unsloth/kernels/moe/grouped_gemm/kernels/tuning.py]
* '''Lines:''' 1-277

=== Key Classes ===
<syntaxhighlight lang="python">
@dataclass
class KernelConfigForward:
    BLOCK_SIZE_M: int = 32
    BLOCK_SIZE_N: int = 32
    BLOCK_SIZE_K: int = 32
    num_warps: int = 4
    num_stages: int = 2

@dataclass
class KernelConfigBackward_dW:
    # Similar configuration for weight gradient kernel
    ...

@dataclass
class KernelConfigBackward_dX:
    # Similar configuration for input gradient kernel
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
    KernelConfigForward,
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
