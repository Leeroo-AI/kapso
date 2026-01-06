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
Loading pretrained models with automatic tensor-parallel sharding across multiple devices for memory-efficient inference and training.

=== Description ===
Tensor Parallel (TP) model loading distributes a model's parameters across multiple devices by partitioning weight matrices along specific dimensions. This principle enables loading models larger than single-device memory by sharding layers like attention projections and feed-forward networks. Each device holds only a fraction of the full model weights, with the distribution strategy (TP plan) determining which parameters to shard and along which dimensions.

The loading process must coordinate weight distribution so that each rank receives its assigned shard while maintaining mathematical equivalence to the full model. For attention mechanisms, this typically means sharding query, key, value, and output projections along the hidden dimension. For feed-forward networks, the first projection is column-sharded and the second is row-sharded to minimize communication.

=== Usage ===
Apply this principle when loading models that exceed single-device memory or when distributing computation for faster training/inference. Specify a device_mesh to define the tensor-parallel topology and a tp_plan (either "auto" for automatic sharding or a detailed dictionary) to control which layers are sharded. The TP size (number of devices) must evenly divide attention head counts and hidden dimensions.

== Theoretical Basis ==
'''Parameter Sharding Strategy:'''

For a weight matrix W ∈ R^(m×n), tensor parallelism distributes it across P devices:

'''Column-wise sharding (Colwise):'''
* W = [W_0, W_1, ..., W_{P-1}] where each W_i ∈ R^(m×n/P)
* Used for attention projections (Q, K, V) and FFN first layer
* Forward: Y_i = X·W_i, then concatenate Y = [Y_0, ..., Y_{P-1}]

'''Row-wise sharding (Rowwise):'''
* W = [[W_0], [W_1], ..., [W_{P-1}]]^T where each W_i ∈ R^(m/P×n)
* Used for attention output and FFN second layer
* Requires AllReduce after local computation

'''TP Plan Structure:'''
<pre>
tp_plan = {
    "style": "colwise" | "rowwise" | "replicate",
    "input_layouts": Placement,
    "output_layouts": Placement,
    "use_local_output": bool
}
</pre>

'''Attention Sharding:'''
For multi-head attention with H heads and dimension d:
* Total parameters: W_q, W_k, W_v ∈ R^(d×d), W_o ∈ R^(d×d)
* With P-way TP: Each device stores d/P dimension worth of heads
* Communication: AllReduce only on output projection

'''Load Distribution:'''
<pre>
Given pretrained weights W_full:
1. Parse tp_plan to determine shard strategy per layer
2. For each parameter tensor:
   a. Identify sharding dimension (0 for row, 1 for column)
   b. Calculate shard boundaries: start = rank * (size / P)
   c. Extract local shard: W_local = W_full[start:end, :]
   d. Wrap as DTensor with appropriate placements
3. Load optimizer states if resuming training
</pre>

'''Memory Efficiency:'''
* Single device: M_full = model_size
* With P-way TP: M_per_device ≈ model_size / P
* Additional overhead: activations + optimizer states

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_TensorParallel_from_pretrained]]
