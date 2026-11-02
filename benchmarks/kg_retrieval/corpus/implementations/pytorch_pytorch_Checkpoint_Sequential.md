# Implementation: Checkpoint_Sequential

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|torch.utils.checkpoint|https://pytorch.org/docs/stable/checkpoint.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Function in torch.utils.checkpoint that runs a sequence of modules in segments and recomputes each segment's internals while gradients are being computed.

=== Description ===

checkpoint_sequential divides a nn.Sequential, or a list of modules, into as many chunks as requested. Only the input of each chunk is retained; everything computed inside one is released and regenerated when gradients are needed.

=== Usage ===

Use it to wrap a long chain of identical layers when the model is expressed as a sequential container.

== Code Reference ==

<syntaxhighlight lang="python">
torch.utils.checkpoint.checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs)
</syntaxhighlight>

* ''functions'': a nn.Sequential or list of modules run in order
* ''segments'': number of chunks the sequence is split into
* ''use_reentrant'': selects the autograd implementation; pass it explicitly

== I/O Contract ==

Takes the input tensors of the first module and returns the output of the last, exactly as running the sequence directly would. Gradients are identical up to nondeterminism. Randomness such as dropout is replayed with the saved random state when preserve_rng_state is left enabled.

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from torch.utils.checkpoint import checkpoint_sequential

layers = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024) for _ in range(24)])
x = torch.randn(8, 1024, requires_grad=True)
out = checkpoint_sequential(layers, 4, x, use_reentrant=False)
out.sum().backward()
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:pytorch_pytorch_Single_24GB_Consumer_GPU]]
* [[uses_heuristic::Heuristic:pytorch_pytorch_Checkpoint_Every_Transformer_Block]]
