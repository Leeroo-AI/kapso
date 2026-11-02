# Principle: Data_Parallelism

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PyTorch Distributed|https://pytorch.org/docs/stable/distributed.html]]
* [[source::Paper|PyTorch Distributed: Experiences on Accelerating Data Parallel Training|https://arxiv.org/abs/2006.15704]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Scheme that keeps a complete copy of the model on every device, feeds each device a different slice of the global batch, and averages gradients across devices after each backward pass.

=== Description ===

Every replica computes gradients on its own data, then an all-reduce makes the gradients identical everywhere so the replicas apply the same update and remain in lockstep. Throughput scales almost linearly while the model and its optimizer state fit on one device.

=== Usage ===

Use it as the default way to scale training when a single device can hold the entire model state.

== Theoretical Basis ==

With N replicas and local gradients g_i, each step computes g = (1/N) * sum(g_i) by a ring all-reduce, which moves about 2 * (N-1)/N times the gradient size per device regardless of N. Memory per device is not reduced: it stays equal to the single-device requirement.

== Related Pages ==

* [[implemented_by::Implementation:pytorch_pytorch_DDP_Wrapper]]
