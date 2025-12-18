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
Aggregating gradients across distributed processes to ensure consistent parameter updates in data-parallel training.

=== Description ===
Gradient synchronization ensures that all replicas in a data-parallel group compute the same parameter updates by averaging gradients across devices. After each rank computes gradients on its local batch, the gradients must be synchronized through collective communication before the optimizer step. This principle is fundamental to data parallelism, enabling mathematically equivalent training to single-device training with a larger batch size.

The synchronization can happen automatically (via DDP/FSDP hooks) or manually (via explicit all_reduce calls). In 3D parallelism, gradient synchronization must occur selectively across the data-parallel and context-parallel dimensions, but not across tensor-parallel dimensions where gradients are already appropriately sharded or local.

=== Usage ===
Apply gradient synchronization after backward pass and before optimizer step in data-parallel training. In 3D parallelism with manual gradient management, use all_reduce across the dp_cp flattened mesh. When using FSDP with NO_SHARD or DDP, synchronization happens automatically through registered backward hooks.

== Theoretical Basis ==
'''Gradient Aggregation:'''

For N data-parallel replicas computing local gradients g_i:

'''AllReduce Operation:'''
* Input: Each rank i has gradient g_i ∈ R^d
* Output: All ranks receive ḡ = (1/N) Σ g_i
* Communication: Ring-AllReduce or Tree-AllReduce
* Complexity: O(d) per device with optimal algorithms

'''Mathematical Equivalence:'''
<pre>
Single-device with batch B:
  L = (1/B) Σ_{j=1}^B ℓ(x_j, θ)
  ∇L = (1/B) Σ_{j=1}^B ∇ℓ(x_j, θ)

Data-parallel with N devices, local batch b = B/N:
  L_i = (1/b) Σ_{j ∈ B_i} ℓ(x_j, θ)
  ∇L_i = (1/b) Σ_{j ∈ B_i} ∇ℓ(x_j, θ)

  Global gradient:
  ∇L = (1/N) Σ_{i=1}^N ∇L_i = (1/B) Σ_{j=1}^B ∇ℓ(x_j, θ)

Equivalence requires: ḡ = (1/N) Σ g_i
</pre>

'''AllReduce Implementation:'''
<pre>
Ring-AllReduce for N ranks:
1. Chunk gradients into N parts
2. For step s in [0, N-1]:
     Send chunk s to rank (i+1) mod N
     Receive chunk from rank (i-1) mod N
     Accumulate received chunk
3. For step s in [0, N-1]:
     Send accumulated chunk to rank (i+1) mod N
     Receive final chunk from rank (i-1) mod N

Total communication: 2(N-1)/N × d ≈ 2d per device
</pre>

'''ReduceOp Types:'''
* SUM: ḡ = Σ g_i (then manually divide by N)
* AVG: ḡ = (1/N) Σ g_i (automatic averaging)
* MAX/MIN: For special use cases

'''3D Parallelism Integration:'''

In mesh (DP, TP, CP):

'''Gradient States by Dimension:'''
* TP dimension: Gradients already local/sharded, no sync needed
* DP dimension: Different data → Must synchronize
* CP dimension: Different sequence chunks → Must synchronize

'''Synchronization Strategy:'''
<pre>
# Option 1: FSDP handles DP automatically
if using FSDP:
    # Only sync manually across CP
    for param in model.parameters():
        if param.grad is not None and cp_size > 1:
            all_reduce(param.grad, group=cp_mesh.get_group())

# Option 2: Manual sync across DP+CP
if manual_gradient_management:
    dp_cp_mesh = world_mesh["dp", "cp"]._flatten()
    for param in model.parameters():
        if param.grad is not None:
            all_reduce(param.grad, group=dp_cp_mesh.get_group())
</pre>

'''DTensor Gradient Handling:'''
When parameters are DTensors (from tensor parallelism):
<pre>
if isinstance(param.grad, DTensor):
    # Convert to local tensor for cross-mesh communication
    local_grad = param.grad.to_local()

    # All-reduce across dp_cp
    dist.all_reduce(local_grad, group=dp_cp_group)

    # Average
    local_grad = local_grad / dp_cp_size

    # Convert back to DTensor
    param.grad = DTensor.from_local(
        local_grad,
        device_mesh=param.grad.device_mesh,
        placements=param.grad.placements
    )
</pre>

'''Communication Volume:'''
For model with M parameters:
* Per rank communication: 2M (send + receive)
* Total network traffic: 2M × N
* Bandwidth utilization: Critical for scaling

'''Gradient Accumulation:'''
When using gradient accumulation steps G:
<pre>
for micro_batch in range(G):
    loss = model(micro_batch)
    loss = loss / G  # Scale loss
    loss.backward()  # Accumulate gradients

# Synchronize only after all micro-batches
all_reduce_gradients()  # Single sync for all accumulated grads
optimizer.step()
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_AllReduce_gradients]]
