# Implementation: Grouped_GEMM_Forward

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
Triton kernels for grouped GEMM forward pass with MoE-specific fusions.

=== Description ===
This module implements the forward pass Triton kernels for grouped GEMM operations. It supports various fusions for MoE workloads including input permutation (permute_x), output permutation (permute_y), and TMA (Tensor Memory Accelerator) operations on SM90+ GPUs.

=== Usage ===
Called by the grouped_gemm_forward interface function for MoE layer computations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/kernels/forward.py unsloth/kernels/moe/grouped_gemm/kernels/forward.py]
* '''Lines:''' 1-265

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.kernels.forward import (
    _grouped_gemm_forward_kernel,
    _autotuned_grouped_gemm_forward_kernel,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
