# Principle: Fully_Sharded_Data_Parallel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|FullyShardedDataParallel|https://pytorch.org/docs/stable/fsdp.html]]
* [[source::Paper|ZeRO: Memory Optimizations Toward Training Trillion Parameter Models|https://arxiv.org/abs/1910.02054]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Variant of data parallelism that partitions parameters, gradients and optimizer state across all ranks and gathers each layer's weights only while that layer is being computed.

=== Description ===

Instead of replicating state, each rank owns one shard of it. Before a layer runs, its full weights are assembled from all shards; after use they are released. Gradients are reduced and scattered so each rank updates only its own shard. Per-device state shrinks roughly in proportion to the number of ranks, at the price of extra gather and scatter communication.

=== Usage ===

Choose it when model state, not the batch, is what overflows a single device, and fast interconnect is available.

== Theoretical Basis ==

Per-device memory for state falls from 16P bytes (mixed precision with Adam, P parameters) to about 16P/N plus the largest gathered unit. Communication volume per step is about 1.5 times that of plain data parallelism: one gather in forward, one in backward and one reduce-scatter.

== Related Pages ==

* [[implemented_by::Implementation:pytorch_pytorch_FSDP_Wrapper]]
