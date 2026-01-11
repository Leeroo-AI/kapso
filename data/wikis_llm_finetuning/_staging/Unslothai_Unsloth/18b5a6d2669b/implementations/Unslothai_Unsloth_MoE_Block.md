# Implementation: MoE_Block

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
Generic MoE block implementation serving as a base for model-specific MoE layers.

=== Description ===
This module provides a generic MoE block implementation that can be subclassed for specific model architectures. It handles common MoE patterns like expert selection, routing, and output aggregation.

=== Usage ===
Used as a base class for model-specific MoE implementations like Llama4MoELayer and Qwen3MoELayer.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/moe/grouped_gemm/reference/moe_block.py unsloth/kernels/moe/grouped_gemm/reference/moe_block.py]
* '''Lines:''' 1-161

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.moe.grouped_gemm.reference.moe_block import MoEBlock
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
