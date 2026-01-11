# Implementation: Qwen3_MoE_Layer

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Models]], [[domain::MoE]], [[domain::Reference]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Reference implementation for Qwen 3 style Mixture-of-Experts layer using grouped GEMM.

=== Description ===
This module provides a reference implementation of the Qwen 3 MoE layer architecture. It includes the expert routing logic and grouped GEMM integration specific to Qwen 3 MoE design patterns.

=== Usage ===
Used as a reference/baseline for testing grouped GEMM kernels against expected Qwen 3 MoE behavior.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py]
* '''Lines:''' 1-348

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.reference.layers.qwen3_moe import (
    Qwen3MoELayer,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
