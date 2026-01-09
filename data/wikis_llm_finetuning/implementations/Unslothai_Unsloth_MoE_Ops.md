# Implementation: MoE_Ops

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Models]], [[domain::MoE]], [[domain::Operations]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Low-level MoE routing operations including top-k selection and token-expert assignment.

=== Description ===
This module provides primitive operations used in MoE routing, including top-k expert selection, computing token-to-expert assignments, and aggregating expert outputs with routing weights.

=== Usage ===
Called by MoE layers for computing expert routing and aggregating results.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py]
* '''Lines:''' 1-151

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import (
    compute_expert_assignment,
    aggregate_expert_outputs,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
