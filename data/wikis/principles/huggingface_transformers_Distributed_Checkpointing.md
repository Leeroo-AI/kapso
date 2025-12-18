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
Saving and loading model and optimizer states in distributed training with proper handling of sharded parameters across devices.

=== Description ===
Distributed checkpointing enables saving and restoring training state in parallel training scenarios where model parameters, gradients, and optimizer states are distributed across multiple devices. This principle addresses the challenge that traditional checkpointing (saving from rank 0) becomes a bottleneck and memory constraint as models grow larger. Distributed Checkpoint (DCP) allows each rank to save and load only its local shards concurrently, dramatically reducing I/O time and memory requirements.

The checkpointing process must preserve the distribution strategy metadata so that loading can correctly reconstruct the distributed state, potentially with different parallelism configurations. This includes tracking parameter placements (sharded, replicated), device mesh topology, and optimizer state distribution.

=== Usage ===
Apply distributed checkpointing when training with tensor parallelism, FSDP, or other sharding strategies where the full model doesn't fit in single-device memory. Use PyTorch's Distributed Checkpoint (DCP) API to save and load state_dicts that contain both model and optimizer states. Each rank saves its local shards to separate files, with metadata coordinating the overall checkpoint.

== Theoretical Basis ==
'''Checkpoint State Structure:'''

Complete training state consists of:
<pre>
State = {
    model: {param_name: param_tensor, ...},
    optimizer: {
        state: {param_id: {momentum, variance, ...}},
        param_groups: [{lr, weight_decay, ...}]
    },
    epoch: int,
    step: int,
    rng_state: random_state
}
</pre>

'''Traditional Checkpointing (Centralized):'''
<pre>
Rank 0:
  1. Gather all parameters from all ranks → O(M) memory
  2. Gather all optimizer states → O(2M) for Adam
  3. Save to single file → Sequential I/O bottleneck
  4. Other ranks wait idle

Problems:
  - Rank 0 needs 3M memory (params + optimizer states)
  - Serial save: O(3M / bandwidth) time
  - Load: All ranks wait for single read, then scatter
</pre>

'''Distributed Checkpointing (DCP):'''
<pre>
All ranks simultaneously:
  1. Save local parameter shards → O(M/N) per rank
  2. Save local optimizer states → O(2M/N) per rank
  3. Parallel writes to shared filesystem

Benefits:
  - Each rank: O(3M/N) memory
  - Parallel save: O(3M / (N × bandwidth)) time
  - N× speedup for save/load
</pre>

'''Metadata Preservation:'''

DCP saves distribution metadata:
<pre>
Metadata = {
    device_mesh: {
        mesh_shape: (DP, TP, CP),
        mesh_dim_names: ["dp", "tp", "cp"]
    },
    parameter_placements: {
        "layer.0.weight": [Shard(0)],  # Sharded on dim 0
        "layer.0.bias": [Replicate()],  # Replicated
    },
    optimizer_placements: {...}
}
</pre>

'''State Dict API:'''

Stateful protocol for DCP:
<pre>
class Stateful:
    def state_dict(self) -> dict:
        """Return state dictionary for saving."""
        pass

    def load_state_dict(self, state_dict: dict):
        """Load from state dictionary."""
        pass
</pre>

'''Saving Process:'''
<pre>
1. Create Stateful wrapper:
   state = {"model": model, "optimizer": optimizer}

2. Get distributed state_dict:
   model_state, optim_state = get_state_dict(model, optimizer)
   # Extracts DTensor metadata and local shards

3. Save with DCP:
   dcp.save(
       state_dict={"model": model_state, "optim": optim_state},
       checkpoint_id="checkpoint_dir",
       storage_writer=FileSystemWriter()  # Or custom writer
   )

4. Each rank writes:
   - checkpoint_dir/
     ├── __metadata__  (rank 0 writes coordination info)
     ├── __0_0.pt      (rank 0's shards)
     ├── __1_0.pt      (rank 1's shards)
     └── ...
</pre>

'''Loading Process:'''
<pre>
1. Create model/optimizer with same or different parallelism

2. Get empty state_dict structure:
   model_state, optim_state = get_state_dict(model, optimizer)

3. Load with DCP:
   dcp.load(
       state_dict={"model": model_state, "optim": optim_state},
       checkpoint_id="checkpoint_dir"
   )
   # Each rank reads only its required shards

4. Apply loaded state:
   set_state_dict(
       model, optimizer,
       model_state_dict=model_state,
       optim_state_dict=optim_state
   )
</pre>

'''Resharding on Load:'''

DCP supports loading with different parallelism:
<pre>
Save: TP=2, DP=4 (8 ranks)
Load: TP=4, DP=2 (8 ranks)

DCP automatically:
  1. Reads metadata from checkpoint
  2. Determines resharding plan
  3. Each rank reads necessary source shards
  4. Reshards to target distribution
</pre>

'''Integration with 3D Parallelism:'''
<pre>
For mesh (DP=2, TP=2, CP=2):

Parameter distribution:
  - TP-sharded: Each TP rank saves different shard
  - DP-replicated: All DP ranks save identical copy
  - CP-replicated: All CP ranks save identical copy

Optimization:
  - DCP deduplicates replicated parameters
  - Only one replica saved per DP group
  - Only one replica saved per CP group
  - Result: Efficient storage, no redundancy
</pre>

'''Storage Writers:'''
<pre>
FileSystemWriter: Standard filesystem (NFS, Lustre)
  - Concurrent writes from all ranks
  - Requires parallel filesystem

TensorStoreWriter: TensorStore backend
  - Optimized for large-scale checkpoints
  - Supports cloud storage (S3, GCS)
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_DCP_save]]
