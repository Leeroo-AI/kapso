{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PyTorch Distributed|https://pytorch.org/docs/stable/distributed.html]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Distributed_Computing]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Replicating models across devices with automatic gradient synchronization for data-parallel training at scale.

=== Description ===
Data parallelism enables training on larger datasets by distributing batches across multiple devices, each holding a complete copy of the model. After computing gradients on their local batch, devices synchronize gradients through collective communication before updating parameters. This principle encompasses various strategies for managing the model replicas and gradient synchronization, from simple data-parallel to fully sharded data parallelism (FSDP).

FSDP extends basic data parallelism by sharding model parameters, gradients, and optimizer states across devices while maintaining the data-parallel training semantics. During forward and backward passes, parameters are gathered just-in-time, computed with, then discarded to minimize memory footprint. This allows training models that don't fit in single-device memory even when considering all model states.

=== Usage ===
Apply data parallelism when training data is large and can be partitioned across devices, and when model parameters can be replicated (or sharded with FSDP) across the data-parallel dimension. In 3D parallelism setups, data parallelism operates orthogonally to tensor and pipeline parallelism, replicating the tensor-parallel sharded model across multiple data-parallel ranks.

== Theoretical Basis ==
'''Basic Data Parallelism:'''

For N devices with global batch B distributed as local batches b_i:
* Each device i holds full model copy θ_i
* Local gradient: g_i = ∇L(θ_i, b_i)
* Synchronized gradient: ḡ = (1/N) Σ g_i via AllReduce
* Update: θ ← θ - α·ḡ (all devices identical)

'''FSDP Sharding Strategy:'''

Parameters θ split into N shards: θ = {θ_0, θ_1, ..., θ_{N-1}}
* Each device i owns shard θ_i persistently
* Forward pass: All-Gather(θ) → compute → discard non-local shards
* Backward pass: All-Gather(θ) → compute gradients → Reduce-Scatter(g)

'''Sharding Strategies:'''
<pre>
FULL_SHARD: Shard parameters, gradients, optimizer states
    Memory per device: O(M/N) where M is model size

SHARD_GRAD_OP: Shard only gradients and optimizer states
    Memory per device: O(M + G/N) where G is gradient size

NO_SHARD: Replicate everything (standard DDP)
    Memory per device: O(M)
    Best communication efficiency for gradient sync
</pre>

'''Communication Pattern:'''
<pre>
For FULL_SHARD during training step:
1. Forward:
   - All-Gather parameters for current layer
   - Compute forward
   - Discard non-local parameters

2. Backward:
   - All-Gather parameters for current layer
   - Compute gradients
   - Reduce-Scatter gradients to owning rank
   - Discard non-local parameters

3. Optimizer:
   - Each rank updates only its parameter shard
   - No additional communication needed
</pre>

'''Memory Efficiency:'''
* Model parameters: M/N per device (FULL_SHARD)
* Gradients: G/N per device (sharded)
* Optimizer states: 2M/N per device (Adam: momentum + variance)
* Peak memory: 4M/N (parameter + gradient + 2 optimizer states)

'''Integration with Tensor Parallelism:'''
In 3D parallelism, data parallelism operates on tensor-parallel model replicas:
* TP dimension: Model sharded across TP devices
* DP dimension: TP-sharded model replicated across DP devices
* Each DP rank holds identical TP shards
* Gradient sync only across DP dimension (TP gradients already local)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_FSDP_wrapping]]
