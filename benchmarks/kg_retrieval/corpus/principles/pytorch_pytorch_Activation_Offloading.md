# Principle: Activation_Offloading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Paper|ZeRO-Offload: Democratizing Billion-Scale Model Training|https://arxiv.org/abs/2101.06840]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Technique that copies tensors kept for backpropagation to pinned CPU memory during the forward pass and prefetches them to the accelerator just before the backward pass needs them.

=== Description ===

Instead of discarding intermediate results, activation offloading moves them across the host interconnect. Transfers are issued on a separate stream so they overlap with computation, and the backward pass finds its inputs already resident again. No computation is repeated, so the cost is interconnect bandwidth and the amount of host memory available.

=== Usage ===

Consider it when the host has ample memory and a fast link to the device, and when throughput matters more than the extra compute that recomputation would add.

== Theoretical Basis ==

Device memory held by intermediate tensors is reduced by the fraction moved to the host. A tensor of size s moved over a link with bandwidth B costs s/B each way; the offload is free of slowdown only if that time hides behind computation of neighbouring layers. The technique therefore pays off when arithmetic intensity per layer is high relative to the size of its outputs.

== Related Pages ==

* [[implemented_by::Implementation:pytorch_pytorch_Save_On_CPU_Hook]]
