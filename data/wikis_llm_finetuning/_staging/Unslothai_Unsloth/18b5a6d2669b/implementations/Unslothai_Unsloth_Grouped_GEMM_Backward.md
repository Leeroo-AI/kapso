# Implementation: Grouped_GEMM_Backward

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::MoE]], [[domain::GEMM]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Triton kernels for grouped GEMM backward passes (dW and dX) used in MoE training.

=== Description ===
This module implements the backward pass kernels for grouped GEMM operations. It includes separate kernels for computing weight gradients (dW) and input gradients (dX), with TMA support for SM90+ GPUs.

=== Usage ===
Called automatically during backward pass of MoE layers when using grouped GEMM autograd functions.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/kernels/backward.py unsloth/kernels/moe/grouped_gemm/kernels/backward.py]
* '''Lines:''' 1-502

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.kernels.backward import (
    _grouped_gemm_dW_kernel,
    _grouped_gemm_dX_kernel,
    _autotuned_grouped_gemm_dW_kernel,
    _autotuned_grouped_gemm_dX_kernel,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
