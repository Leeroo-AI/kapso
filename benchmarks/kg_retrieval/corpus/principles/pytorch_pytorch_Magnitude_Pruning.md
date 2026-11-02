# Principle: Magnitude_Pruning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Learning both Weights and Connections for Efficient Neural Networks|https://arxiv.org/abs/1506.02626]]
* [[source::Doc|PyTorch pruning tutorial|https://pytorch.org/tutorials/intermediate/pruning_tutorial.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Compression]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Compression technique that removes the connections judged least important by weight magnitude, producing sparse tensors that are then fine-tuned to recover accuracy.

=== Description ===

Weights near zero contribute little to a layer's output, so setting them to zero changes the function only slightly. Pruning can be applied once or gradually, per layer or across the whole network, and the surviving weights are retrained for a few epochs.

=== Usage ===

Use it to reduce parameter count or exploit sparse kernels, accepting that unstructured sparsity speeds up inference only on supporting hardware.

== Theoretical Basis ==

Given sparsity target s, rank weights by |w| within the chosen scope and zero the lowest fraction s. A binary mask is stored so the zeros stay fixed during fine-tuning.

== Related Pages ==

* [[implemented_by::Implementation:pytorch_pytorch_L1_Unstructured_Prune]]
