# Implementation: Pipeline_Schedule

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|torch.distributed.pipelining|https://pytorch.org/docs/stable/distributed.pipelining.html]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Schedule classes in torch.distributed.pipelining, such as ScheduleGPipe and Schedule1F1B, that order micro-batch forward and backward steps across the stages of a pipeline.

=== Description ===

A stage object wraps one slice of the model; a schedule object drives it. ScheduleGPipe runs all forwards then all backwards, while Schedule1F1B alternates them to limit stored activations.

=== Usage ===

Create the stage for the local rank, then step the schedule with a batch split into micro-batches.

== Code Reference ==

<syntaxhighlight lang="python">
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
stage = PipelineStage(submodule, stage_index, num_stages, device)
schedule = ScheduleGPipe(stage, n_microbatches=8, loss_fn=loss_fn)
</syntaxhighlight>
