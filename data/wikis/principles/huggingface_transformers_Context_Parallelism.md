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
Partitioning sequence length across devices to enable training with context windows larger than single-device memory capacity.

=== Description ===
Context parallelism (CP), also known as sequence parallelism, distributes the sequence dimension of input tensors across multiple devices, allowing training with extremely long sequences that wouldn't fit in single-device memory. Unlike tensor parallelism which shards model parameters, context parallelism shards the input sequence and intermediate activations while keeping model parameters replicated (or tensor-parallel sharded) across CP ranks.

This principle requires careful coordination of attention mechanisms, as each device computes attention only over its local sequence chunk. Ring attention or similar techniques enable computing attention across all sequence chunks without materializing the full attention matrix. The context parallel context manager handles automatic sharding of specified buffers along their sequence dimensions and coordinates the necessary communication for attention computation.

=== Usage ===
Apply context parallelism when training or fine-tuning models with very long context windows (e.g., 32K-128K tokens) that exceed memory capacity. Use the context_parallel context manager during forward and backward passes, specifying which tensors (input_ids, labels, position_ids) should be sharded along the sequence dimension. CP works orthogonally to TP and DP dimensions in 3D parallelism.

== Theoretical Basis ==
'''Sequence Partitioning:'''

For sequence length L and P context-parallel ranks:
* Full sequence: X ∈ R^(B×L×D) where B=batch, D=hidden_dim
* Each rank holds chunk: X_i ∈ R^(B×L/P×D)
* Concatenation: [X_0, X_1, ..., X_{P-1}] = X

'''Ring Attention Mechanism:'''
Standard attention: A = softmax(QK^T/√d)V requires O(L²) memory

With ring attention across P ranks:
<pre>
For each rank i:
  1. Local attention within chunk:
     A_ii = softmax(Q_i K_i^T / √d) V_i

  2. Ring communication for cross-chunk attention:
     For j in [1, P-1]:
       - Send K_i, V_i to rank (i+1) mod P
       - Receive K_{i-j}, V_{i-j} from rank (i-1) mod P
       - Compute A_i,i-j = softmax(Q_i K_{i-j}^T / √d) V_{i-j}

  3. Aggregate: A_i = Σ_j A_ij
</pre>

Memory reduction: O(L²/P) per device

'''Buffer Sharding:'''
The context_parallel manager automatically shards specified buffers:
<pre>
Input: buffer ∈ R^(B×L×...)
Sharded: buffer_i ∈ R^(B×L/P×...)

buffer_seq_dims parameter specifies which dimension is sequence:
  [1, 1, 1] → second dimension (typical for [batch, seq, hidden])
</pre>

'''Gradient Flow:'''
During backward pass:
* Gradients computed on local sequence chunks
* Ring communication in reverse for attention gradients
* Each rank accumulates gradients for its sequence chunk
* Final gradients sharded along sequence dimension

'''Communication Pattern:'''
<pre>
Forward pass:
  - Input tensors: Scatter sequence dimension
  - Attention: Ring All-to-All for K, V
  - Output: Keep sharded (each rank has L/P)

Backward pass:
  - Gradient input: Sharded (L/P per rank)
  - Attention gradients: Ring All-to-All in reverse
  - Gradient output: Keep sharded

No AllReduce needed across CP dimension for parameters
(CP replicates parameters, unlike TP which shards them)
</pre>

'''Integration with 3D Parallelism:'''
<pre>
Mesh: (DP, TP, CP)

Parameter distribution:
  - TP dimension: Sharded
  - DP dimension: Replicated (or FSDP-sharded)
  - CP dimension: Replicated

Activation distribution:
  - Batch dimension: Replicated within DP rank
  - Sequence dimension: Sharded across CP
  - Hidden dimension: Sharded across TP (if sequence parallel)

Example: 8 GPUs, DP=2, TP=2, CP=2, sequence_length=4096
  Each rank holds: 2048 sequence tokens (4096/2)
  TP ranks: Share same sequence chunks, different param shards
  DP ranks: Different data, same model
</pre>

'''Position Embeddings:'''
Must handle position IDs correctly for sharded sequences:
<pre>
Full position_ids: [0, 1, 2, ..., L-1]
CP rank i: [i*L/P, i*L/P+1, ..., (i+1)*L/P-1]

Example: L=1024, P=4
  Rank 0: [0, 1, ..., 255]
  Rank 1: [256, 257, ..., 511]
  Rank 2: [512, 513, ..., 767]
  Rank 3: [768, 769, ..., 1023]
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Context_parallel_execution]]
