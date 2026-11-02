# Principle: Tensor_Parallelism

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Megatron-LM|https://github.com/NVIDIA/Megatron-LM]]
* [[source::Paper|Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism|https://arxiv.org/abs/1909.08053]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Technique that divides the weight matrices of individual layers among devices, so each computes a block of the matrix product and the pieces are combined with collective communication.

=== Description ===

In a transformer feed-forward block, the first matrix is divided along its output dimension and the second along its input dimension, so the nonlinearity between them needs no communication and only one all-reduce is required per block. Attention heads are likewise distributed across devices.

=== Usage ===

Use it when a layer is too wide for one device, typically inside a fast-interconnect node.

== Theoretical Basis ==

For Y = XA with A = [A_1, A_2], each device computes XA_i independently. For Z = YB with B split by rows, each device computes Y_i B_i and the partial products are summed by an all-reduce. Communication happens in every layer, so the scheme needs high-bandwidth links.

== Related Pages ==

* [[implemented_by::Implementation:NVIDIA_Megatron-LM_ColumnParallelLinear]]
