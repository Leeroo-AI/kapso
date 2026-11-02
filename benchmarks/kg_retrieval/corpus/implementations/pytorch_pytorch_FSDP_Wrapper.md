# Implementation: FSDP_Wrapper

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|FullyShardedDataParallel|https://pytorch.org/docs/stable/fsdp.html]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

FullyShardedDataParallel is the PyTorch wrapper class that distributes a module's parameters, gradients and optimizer state across the processes of a distributed job. Constructing it around a model replaces each wrapped module's parameters with local shards, and a full copy of a unit's weights is gathered for the duration of its forward or backward computation and released afterwards. How the model is divided into those units is decided by the auto_wrap_policy argument, for example a transformer-layer policy that names the block class, while the sharding_strategy argument chooses between FULL_SHARD, SHARD_GRAD_OP, NO_SHARD and hybrid sharding, which determines which tensors remain replicated. Further arguments cover CPU offload, mixed-precision settings, backward prefetching and the device identifier.

=== Description ===

FullyShardedDataParallel is PyTorch's wrapper for sharded data-parallel training: it spreads a model's parameters, gradients and optimizer state over many GPUs so that networks larger than a single device can be trained. Each wrapped unit owns a flat parameter that is sharded evenly, and unit boundaries set the peak memory.

=== Usage ===

Wrap the model once, after the process group is initialised and before the optimizer is created.

== Code Reference ==

<syntaxhighlight lang="python">
torch.distributed.fsdp.FullyShardedDataParallel(module, sharding_strategy=None,
    auto_wrap_policy=None, cpu_offload=None, mixed_precision=None,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE, device_id=None)
</syntaxhighlight>

== Usage Examples ==

<syntaxhighlight lang="python">
policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
model = FSDP(model, auto_wrap_policy=policy, device_id=torch.cuda.current_device())
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:pytorch_pytorch_NCCL_Multi_Node_Cluster]]
