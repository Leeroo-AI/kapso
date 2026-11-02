# Principle: Pipeline_Parallelism

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PyTorch Distributed|https://pytorch.org/docs/stable/distributed.html]]
* [[source::Paper|GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism|https://arxiv.org/abs/1811.06965]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Technique that assigns consecutive groups of layers to different devices and streams micro-batches through them so the stages work concurrently.

=== Description ===

Each device holds one stage. A batch is cut into micro-batches that enter the first stage one after another; while stage two processes the first, stage one already works on the second. The idle time at the start and end of a step, called the bubble, shrinks as the number of micro-batches grows.

=== Usage ===

Use it to span several nodes when links between them are slow, since only boundary activations are sent.

== Theoretical Basis ==

With p stages and m micro-batches, the fraction of time lost to the bubble is (p-1)/(m+p-1). Schedules such as one-forward-one-backward interleave the two passes to cap the number of stored micro-batch activations.

== Related Pages ==

* [[implemented_by::Implementation:pytorch_pytorch_Pipeline_Schedule]]
