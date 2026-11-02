# Implementation: ColumnParallelLinear

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Megatron-LM|https://github.com/NVIDIA/Megatron-LM]]
* [[source::Doc|Megatron Core tensor parallel|https://docs.nvidia.com/megatron-core/developer-guide/latest/]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Linear layer in Megatron-LM whose weight matrix is divided along its output dimension, giving each tensor-parallel rank one block of output features.

=== Description ===

The layer computes Y_i = X A_i on each rank. With gather_output enabled the blocks are concatenated by an all-gather; otherwise the sharded output feeds a following row-parallel layer directly.

=== Usage ===

Use it as the first matrix of a feed-forward or attention projection when layers are split across devices.

== Code Reference ==

<syntaxhighlight lang="python">
megatron.core.tensor_parallel.ColumnParallelLinear(input_size, output_size, *, config,
    init_method, bias=True, gather_output=False, skip_bias_add=False)
</syntaxhighlight>
