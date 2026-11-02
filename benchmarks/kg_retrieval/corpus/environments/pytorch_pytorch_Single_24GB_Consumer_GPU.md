# Environment: Single_24GB_Consumer_GPU

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch|https://github.com/pytorch/pytorch]]
* [[source::Doc|PyTorch Get Started|https://pytorch.org/get-started/locally/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Workstation setup built around a single consumer graphics card with 24 GB of memory, such as an RTX 3090 or RTX 4090, with a recent CUDA toolkit and PyTorch release.

=== Description ===

A 24 GB consumer card offers large compute throughput but no high-speed interconnect and no error-correcting memory. It is the typical target for single-device fine-tuning and prototyping of models up to a few billion parameters.

=== Usage ===

Use this environment description when a technique's memory estimates are quoted for a 24 GB device.

== System Requirements ==

* Linux x86_64, recent NVIDIA driver
* One GPU with 24 GB of device memory
* At least 32 GB of host RAM for data loading

== Dependencies ==

* torch built against a CUDA 12 runtime
* A matching NVIDIA driver; verify with ''nvidia-smi''

== Common Errors ==

* ''CUDA out of memory'' on the first optimizer step usually means optimizer state was not counted in the estimate.
