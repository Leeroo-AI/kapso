# Implementation: Llama4_MoE_Layer

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
Reference implementation for LLaMA 4 style Mixture-of-Experts layer using grouped GEMM.

=== Description ===
This module provides a reference implementation of the LLaMA 4 MoE layer architecture. It includes the expert routing logic, grouped GEMM integration, and weight multiplication patterns specific to the LLaMA 4 MoE design.

=== Usage ===
Used as a reference/baseline for testing grouped GEMM kernels against expected LLaMA 4 MoE behavior.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py]
* '''Lines:''' 1-437

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.reference.layers.llama4_moe import (
    Llama4MoELayer,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
