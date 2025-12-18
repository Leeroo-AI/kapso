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
Partitioning dataset samples across data-parallel ranks to ensure each device processes unique data without overlap.

=== Description ===
Distributed dataset management ensures that in data-parallel training, each rank processes a unique subset of the training data, preventing redundant computation and ensuring proper gradient averaging. This principle involves creating samplers that deterministically assign dataset indices to ranks based on their position in the data-parallel group, accounting for the total number of data-parallel replicas.

The partitioning must be coordinated so that across all ranks, the entire dataset is covered exactly once per epoch (or with controlled overlap/drop strategies). Shuffling, when enabled, must use synchronized random seeds to maintain deterministic partitioning while still randomizing the order within each rank's subset.

=== Usage ===
Apply this principle when setting up DataLoaders for data-parallel training. Create a DistributedSampler with num_replicas equal to the data-parallel world size and rank corresponding to the data-parallel rank. In 3D parallelism, only partition across the data-parallel dimension, not tensor or pipeline parallel dimensions.

== Theoretical Basis ==
'''Dataset Partitioning:'''

Given dataset D with N samples and P data-parallel ranks:

'''Index Assignment:'''
* Rank r receives indices: I_r = {i | i mod P = r, i ∈ [0, N-1]}
* Each rank processes N/P samples per epoch (assuming N divisible by P)
* No overlap: I_i ∩ I_j = ∅ for i ≠ j
* Complete coverage: ∪ I_i = D

'''With Shuffling:'''
<pre>
1. Create shuffled index mapping using synchronized seed:
   π = shuffle([0, 1, ..., N-1], seed=epoch)

2. Partition shuffled indices:
   I_r = {π[r], π[P+r], π[2P+r], ...}

3. Each epoch: Update seed to get new shuffle
   seed = base_seed + epoch_number
</pre>

'''Batch Formation:'''
For global batch size B and local batch size b = B/P:
* Each rank creates batches of size b from its partition
* Global iteration sees B samples across all ranks
* Effective global batch: Concatenation of all local batches

'''Drop Last Behavior:'''
When N is not divisible by P:
* drop_last=True: Drop N mod P samples to ensure equal sizes
* drop_last=False: Pad shorter partitions with repeated samples

'''Integration with 3D Parallelism:'''
<pre>
In mesh topology (DP, TP, CP):
- TP ranks: Share identical data (same indices)
- CP ranks: Share identical data (same indices)
- DP ranks: Disjoint data (partitioned indices)

Example: 8 ranks with DP=2, TP=2, CP=2
  DP rank 0: [TP0, TP1] × [CP0, CP1] → indices [0, 2, 4, 6, ...]
  DP rank 1: [TP0, TP1] × [CP0, CP1] → indices [1, 3, 5, 7, ...]
</pre>

'''Deterministic Training:'''
For reproducibility:
* Use fixed seed for initial shuffle
* Increment seed by epoch for different shuffles each epoch
* All ranks use same seed → deterministic partition
* Setting seed = base_seed + epoch ensures consistency across runs

'''Sampler Logic:'''
<pre>
def get_sample_indices(dataset_size, rank, world_size, shuffle, seed):
    indices = list(range(dataset_size))

    if shuffle:
        random.seed(seed)
        random.shuffle(indices)

    # Partition
    indices = indices[rank::world_size]

    return indices
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_DistributedSampler_usage]]
