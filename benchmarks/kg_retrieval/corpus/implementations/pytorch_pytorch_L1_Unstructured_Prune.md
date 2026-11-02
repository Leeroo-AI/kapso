# Implementation: L1_Unstructured_Prune

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|torch.nn.utils.prune|https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Compression]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Function torch.nn.utils.prune.l1_unstructured, which zeroes a given fraction of a parameter's entries with the lowest L1 magnitude and attaches a mask so the zeros persist.

=== Description ===

The original tensor is renamed to name_orig and a binary buffer name_mask is registered; a forward pre-hook recomputes the effective parameter as their product. Calling prune.remove later makes the zeros permanent and drops the extras.

=== Usage ===

Call it on one module and parameter name, or loop over layers for network-wide sparsity.

== Code Reference ==

<syntaxhighlight lang="python">
torch.nn.utils.prune.l1_unstructured(module, name, amount, importance_scores=None)
</syntaxhighlight>

* ''amount'': a fraction between 0 and 1, or an absolute count of entries to prune

== Usage Examples ==

<syntaxhighlight lang="python">
import torch.nn.utils.prune as prune
prune.l1_unstructured(model.fc, name="weight", amount=0.3)
print(model.fc.weight_mask.sum())
</syntaxhighlight>
