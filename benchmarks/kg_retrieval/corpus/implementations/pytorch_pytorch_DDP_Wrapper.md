# Implementation: DDP_Wrapper

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|PyTorch Distributed|https://pytorch.org/docs/stable/distributed.html]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

DistributedDataParallel wraps a module so that gradients are averaged across processes during backward, keeping one full replica of the model per process.

=== Description ===

Gradients are grouped into buckets and reduced asynchronously as soon as a bucket is ready, overlapping communication with the remaining backward computation.

=== Usage ===

Wrap the model after moving it to its device and initialising the process group.

== Code Reference ==

<syntaxhighlight lang="python">
torch.nn.parallel.DistributedDataParallel(module, device_ids=None, output_device=None,
    broadcast_buffers=True, bucket_cap_mb=None, find_unused_parameters=False)
</syntaxhighlight>

* ''bucket_cap_mb'': bucket size in MiB; a size of 25 MiB is used when it is None

== Usage Examples ==

<syntaxhighlight lang="python">
torch.distributed.init_process_group("nccl")
model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank])
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:pytorch_pytorch_NCCL_Multi_Node_Cluster]]
