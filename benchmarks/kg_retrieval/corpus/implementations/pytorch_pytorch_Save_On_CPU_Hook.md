# Implementation: Save_On_CPU_Hook

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|Autograd saved tensor hooks|https://pytorch.org/docs/stable/autograd.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Context manager in torch.autograd.graph that parks tensors saved for backward in CPU memory and returns them to the accelerator when gradients are computed.

=== Description ===

save_on_cpu installs a pair of saved-tensor hooks: a pack hook that copies each saved tensor to the host, optionally into pinned memory, and an unpack hook that copies it back on demand during backward.

=== Usage ===

Wrap the forward pass of a memory-hungry module; everything autograd saves inside the block is moved.

== Code Reference ==

<syntaxhighlight lang="python">
torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda")
</syntaxhighlight>

* ''pin_memory'': allocate the host copies in page-locked memory so transfers can be asynchronous
* ''device_type'': the accelerator type the tensors live on

== Usage Examples ==

<syntaxhighlight lang="python">
import torch

with torch.autograd.graph.save_on_cpu(pin_memory=True):
    loss = model(batch).sum()
loss.backward()
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:pytorch_pytorch_Single_24GB_Consumer_GPU]]
