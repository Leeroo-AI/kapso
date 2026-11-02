# Principle: Gradient_Checkpointing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Training Deep Nets with Sublinear Memory Cost|https://arxiv.org/abs/1604.06174]]
* [[source::Doc|torch.utils.checkpoint|https://pytorch.org/docs/stable/checkpoint.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Training technique that discards selected intermediate tensors after the forward pass and rebuilds them during backpropagation, lowering peak memory at the cost of extra compute.

=== Description ===

Backpropagation needs the intermediate results of every layer, so activation memory grows linearly with depth. Gradient checkpointing keeps only the tensors at chosen boundaries and, when the backward pass reaches a segment, runs that segment's forward computation again to regenerate what it needs. Peak memory drops sharply while each step costs roughly one additional forward pass.

=== Usage ===

Apply it to deep networks whose stored intermediate results, not their weights, are what exhaust device memory, and where spare compute is cheaper than a larger device.

== Theoretical Basis ==

For a chain of n layers, storing every output costs O(n) memory. Keeping an output every k layers and recomputing the layers in between costs O(n/k + k) memory, minimized near k = sqrt(n), giving O(sqrt(n)) memory. The price is one extra forward evaluation of every layer outside the stored boundaries, which is about a third more compute per step for typical networks.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# forward: keep only boundary tensors
# backward, per segment from the end:
#   rerun the segment's forward from its stored boundary
#   backpropagate through the regenerated tensors
</syntaxhighlight>

== Related Pages ==

* [[implemented_by::Implementation:pytorch_pytorch_Checkpoint_Sequential]]
* [[uses_heuristic::Heuristic:pytorch_pytorch_Checkpoint_Every_Transformer_Block]]
