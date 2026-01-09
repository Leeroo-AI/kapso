# Implementation: Grouped_GEMM_Autotuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::MoE]], [[domain::Autotuning]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Triton autotuning configurations for grouped GEMM kernels used in MoE architectures.

=== Description ===
This module defines autotuning search spaces and configurations for the grouped GEMM Triton kernels. It includes configurations for forward pass, dW (weight gradient), and dX (input gradient) kernels with various block sizes, warp counts, and pipeline stages.

=== Usage ===
Used internally by the grouped GEMM interface when `autotune=True` is passed. Defines the search space Triton explores to find optimal kernel configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py]
* '''Lines:''' 1-396

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.kernels.autotuning import (
    get_autotuning_configs_forward,
    get_autotuning_configs_backward_dW,
    get_autotuning_configs_backward_dX,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
