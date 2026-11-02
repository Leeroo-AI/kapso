# Workflow: Fit_Large_Model_Training_On_One_GPU

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch Examples|https://github.com/pytorch/examples]]
* [[source::Doc|torch.utils.checkpoint|https://pytorch.org/docs/stable/checkpoint.html]]
* [[source::Doc|PyTorch Automatic Mixed Precision|https://pytorch.org/docs/stable/amp.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Recipe for training a network whose working set exceeds a consumer GPU by combining recomputation, reduced-precision arithmetic and micro-batched updates.

=== Description ===

A model that does not fit on a single workstation card can often be trained anyway by attacking the three big memory consumers separately: stored intermediate tensors, parameter and optimizer precision, and the number of samples resident at once. This recipe applies the cheapest remedy first and measures after each change.

=== Usage ===

Follow it when a training script runs out of device memory on a 24 GB card and a larger machine is not available.

== Execution Steps ==

=== Step 1: Measure the budget ===

Record peak device memory for one optimizer step at the smallest useful sample count, and note how much of it is weights, optimizer state and intermediate tensors.

=== Step 2: Move arithmetic to 16-bit ===

Run forward and backward passes in a 16-bit floating-point format, keeping 32-bit master weights. This roughly halves the memory of intermediate tensors and speeds up matrix products on tensor cores.

=== Step 3: Recompute intermediate tensors ===

Mark the largest repeated blocks of the network for recomputation so their intermediate results are rebuilt during the backward pass instead of being stored.

=== Step 4: Restore the update size ===

Process the data in smaller pieces and combine their gradients before each weight update, so the optimization behaves as if the larger group had been used.

=== Step 5: Verify ===

Compare throughput and the loss curve against a reference run on a small subset to confirm the changes did not alter convergence.

== GitHub URL ==

[https://github.com/pytorch/examples pytorch/examples]
