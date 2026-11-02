# Environment: NCCL_Multi_Node_Cluster

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PyTorch Distributed|https://pytorch.org/docs/stable/distributed.html]]
* [[source::Doc|NCCL documentation|https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Distributed_Training]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Group of GPU machines on a shared network with NCCL as the collective-communication backend, a reachable rendezvous endpoint and the same software stack on every host.

=== Description ===

Collective operations need every process to agree on a coordinator address and rank before the first all-reduce. Differences in library versions or blocked ports are the usual cause of stalled jobs.

=== Usage ===

Check it before launching a job that spans more than one host.

== System Requirements ==

* Hosts with NVIDIA GPUs and a fast fabric (InfiniBand or high-speed Ethernet)
* A TCP port range open between all hosts
* Identical driver, CUDA, NCCL and torch versions everywhere

== Dependencies ==

* Launch with torchrun or an equivalent, setting MASTER_ADDR, MASTER_PORT, WORLD_SIZE and RANK for each process
* Set NCCL_SOCKET_IFNAME when a host has several network interfaces

== Common Errors ==

* A watchdog timeout in the first collective: a rank never joined, often because of a closed port or wrong interface.
* ''unhandled system error'': inspect the output with NCCL_DEBUG=INFO.
