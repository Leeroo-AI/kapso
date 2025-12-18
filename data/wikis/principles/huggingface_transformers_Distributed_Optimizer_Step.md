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
Updating model parameters in a distributed training environment after gradient synchronization and clipping.

=== Description ===
The distributed optimizer step principle encompasses updating parameters across multiple devices in parallel training, ensuring that all replicas remain synchronized after the update. In data-parallel training, each rank applies the same optimizer update to its local parameter copy using synchronized gradients. In tensor-parallel training, each rank updates only its parameter shard using locally computed or properly reduced gradients.

The optimizer step must occur after gradient synchronization (for data parallelism) and gradient clipping. In 3D parallelism, the step operates on parameters that may be simultaneously sharded across TP dimensions, replicated/sharded across DP dimensions, and replicated across CP dimensions. Proper coordination ensures all ranks update their parameter portions consistently.

=== Usage ===
Apply the optimizer step after backward pass, gradient synchronization, and gradient clipping. The standard optimizer.step() call works transparently in distributed settings when gradients are properly synchronized. For FSDP, the optimizer automatically handles sharded parameters. For tensor-parallel models, each rank updates its local parameter shards.

== Theoretical Basis ==
'''Parameter Update Rule:'''

Standard SGD with momentum:
<pre>
m_t = β·m_{t-1} + (1-β)·∇L_t       # Momentum
θ_t = θ_{t-1} - α·m_t              # Parameter update
</pre>

Adam optimizer:
<pre>
m_t = β_1·m_{t-1} + (1-β_1)·∇L_t           # First moment
v_t = β_2·v_{t-1} + (1-β_2)·(∇L_t)²        # Second moment
m̂_t = m_t / (1-β_1^t)                      # Bias correction
v̂_t = v_t / (1-β_2^t)
θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)        # Update
</pre>

'''Distributed Parameter Update:'''

'''Data Parallelism:'''
* All ranks have identical parameters: θ_i = θ for all i
* After gradient sync: ∇L_i = ∇L for all i
* Each rank performs identical update: θ_i ← θ_i - α·∇L
* Result: All ranks remain synchronized

'''Tensor Parallelism:'''
* Parameters sharded: θ = [θ_0, θ_1, ..., θ_{P-1}]
* Gradients local/sharded: ∇L = [∇L_0, ∇L_1, ..., ∇L_{P-1}]
* Each rank updates its shard: θ_i ← θ_i - α·∇L_i
* No communication needed (already have correct gradient shard)

'''FSDP (Fully Sharded Data Parallel):'''
* Parameters sharded across DP: θ = [θ_0, θ_1, ..., θ_{N-1}]
* Gradients reduced to owning rank: rank i has ∇L_i
* Optimizer states sharded: m_i, v_i on rank i
* Each rank updates only its shard: θ_i ← θ_i - α·f(∇L_i, m_i, v_i)
* Result: Sharded parameters updated, no All-Gather until next forward

'''Gradient Clipping:'''

Before optimizer step, clip gradients to prevent instability:
<pre>
# Compute global gradient norm
total_norm = √(Σ ||∇θ_i||²)

# In distributed setting, need to aggregate norms
local_norm² = Σ ||∇θ_local||²
global_norm² = AllReduce(local_norm²)
total_norm = √global_norm²

# Clip if exceeds threshold
clip_coef = max_norm / (total_norm + ε)
if clip_coef < 1:
    ∇θ ← clip_coef · ∇θ
</pre>

'''3D Parallelism Optimizer State:'''
<pre>
For parameter θ with dimensions (TP, DP, CP):

Parameter storage:
  - TP dimension: Sharded
  - DP dimension: Replicated or FSDP-sharded
  - CP dimension: Replicated

Optimizer state (momentum, variance):
  - Follows parameter distribution
  - TP-sharded params → TP-sharded states
  - DP-replicated params → DP-replicated states
  - FSDP-sharded params → FSDP-sharded states

Memory per rank:
  - Parameters: M / TP_size (if FSDP: also / DP_size)
  - Optimizer states: 2M / TP_size (Adam)
  - Total: 3M / TP_size per rank
</pre>

'''Closure-based Optimization:'''

Some optimizers (like LBFGS) require re-evaluation:
<pre>
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    return loss

optimizer.step(closure)  # May call closure multiple times
</pre>

'''Learning Rate Scheduling:'''

In distributed training, all ranks must use same LR:
<pre>
scheduler = get_linear_schedule_with_warmup(optimizer, ...)

for epoch in epochs:
    for batch in dataloader:
        # Training step
        optimizer.step()
        scheduler.step()  # Update LR on all ranks

    # All ranks have identical LR
    assert all_ranks_have_same_lr()
</pre>

'''Parameter Update Verification:'''
<pre>
After optimizer.step():

1. Data-parallel dimension:
   - All DP ranks should have identical parameters
   - Verify: AllGather params, compare across DP

2. Tensor-parallel dimension:
   - Each TP rank has different shard
   - Verify: Local shard updated (not identical across TP)

3. Context-parallel dimension:
   - All CP ranks should have identical parameters
   - Verify: AllGather params, compare across CP
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Optimizer_step]]
