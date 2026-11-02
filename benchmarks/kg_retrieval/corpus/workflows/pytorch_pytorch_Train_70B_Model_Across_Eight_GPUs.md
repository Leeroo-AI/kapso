# Workflow: Train_70B_Model_Across_Eight_GPUs

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|TorchTitan|https://github.com/pytorch/torchtitan]]
* [[source::Doc|PyTorch Distributed|https://pytorch.org/docs/stable/distributed.html]]
* [[source::Doc|FullyShardedDataParallel|https://pytorch.org/docs/stable/fsdp.html]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Recipe for pretraining or fine-tuning a seventy-billion-parameter network on one eight-accelerator node by sharding model state and combining parallelism strategies.

=== Description ===

At 70B parameters, weights plus gradients plus optimizer state exceed the memory of any single accelerator by an order of magnitude. The recipe distributes that state first, then adds further splitting only if one node's memory is still not enough.

=== Usage ===

Follow it when a model is too large to replicate on every device of an eight-accelerator node.

== Execution Steps ==

=== Step 1: Estimate the state ===

Add up the bytes of parameters, gradients and optimizer moments at the chosen precision and compare the total with the aggregate memory of the node.

=== Step 2: Shard the model state ===

Partition parameters, gradients and optimizer state across all eight processes, gathering a layer's weights only while it executes.

=== Step 3: Shrink what remains ===

Recompute intermediate tensors for the largest blocks and use a 16-bit compute format to fit the working set.

=== Step 4: Add model splitting if needed ===

If a single layer's weights still do not fit during execution, split the matrices of each layer across devices in addition to sharding.

=== Step 5: Tune communication ===

Overlap gradient and weight transfers with computation and verify scaling efficiency from two to eight devices.

== GitHub URL ==

[https://github.com/pytorch/torchtitan pytorch/torchtitan]
