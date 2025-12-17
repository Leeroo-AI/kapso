= Data Partition Implementation =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || examples/offline_inference/data_parallel.py
|-
| Domains || Data Processing, Algorithm Implementation
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Data Partition Implementation provides the concrete algorithm for distributing prompts across data parallel workers in vLLM. This implementation ensures balanced, deterministic partitioning using a floor-and-remainder strategy.

== Description ==
The implementation partitions a list of prompts into disjoint subsets, one for each data parallel worker. The algorithm:

* Computes a base size (floor division)
* Distributes remainder items to first K workers
* Calculates start/end indices for each worker
* Extracts the partition slice
* Handles edge case of empty partitions

This is a reference implementation used in vLLM's data parallel example and can be adapted for custom use cases.

== Code Reference ==

=== Source Location ===
* '''File''': <code>/tmp/praxium_repo_583nq7ea/examples/offline_inference/data_parallel.py</code>
* '''Lines''': 143-165

=== Implementation ===
<syntaxhighlight lang="python">
# Sample prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 100  # 400 total prompts

# with DP, each rank should process different prompts.
# usually all the DP ranks process a full dataset,
# and each rank processes a different part of the dataset.
floor = len(prompts) // dp_size
remainder = len(prompts) % dp_size

# Distribute prompts into even groups.
def start(rank):
    return rank * floor + min(rank, remainder)

prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]

if len(prompts) == 0:
    # if any rank has no prompts to process,
    # we need to set a placeholder prompt
    prompts = ["Placeholder"]

print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")
</syntaxhighlight>

=== Import ===
No specific import needed - pure Python list slicing.

== I/O Contract ==

=== Input ===
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| prompts || list[str] || Full list of prompts to partition
|-
| global_dp_rank || int || Worker rank (0 to dp_size-1)
|-
| dp_size || int || Total number of workers
|}

=== Output ===
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| prompts || list[str] || Subset of prompts for this worker (may be placeholder)
|}

=== Algorithm Complexity ===
* '''Time''': O(1) for index calculation, O(k) for list slice where k = partition size
* '''Space''': O(k) for partition storage
* '''Distribution''': Max difference between partitions is 1 item

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
# Full dataset
prompts = ["prompt_{}".format(i) for i in range(100)]

# Worker configuration
global_dp_rank = 0
dp_size = 4

# Partition calculation
floor = len(prompts) // dp_size  # 25
remainder = len(prompts) % dp_size  # 0

def start(rank):
    return rank * floor + min(rank, remainder)

# Extract partition
my_prompts = prompts[start(global_dp_rank):start(global_dp_rank + 1)]

print(f"Worker {global_dp_rank}: {len(my_prompts)} prompts")
# Output: Worker 0: 25 prompts
</syntaxhighlight>

=== With Uneven Distribution ===
<syntaxhighlight lang="python">
# Dataset not evenly divisible
prompts = ["prompt_{}".format(i) for i in range(42)]

global_dp_rank = 0
dp_size = 5

floor = len(prompts) // dp_size  # 8
remainder = len(prompts) % dp_size  # 2

def start(rank):
    return rank * floor + min(rank, remainder)

# Worker 0: start(0)=0, start(1)=9, length=9
# Worker 1: start(1)=9, start(2)=18, length=9
# Worker 2: start(2)=18, start(3)=26, length=8
# Worker 3: start(3)=26, start(4)=34, length=8
# Worker 4: start(4)=34, start(5)=42, length=8

my_prompts = prompts[start(global_dp_rank):start(global_dp_rank + 1)]
print(f"Worker {global_dp_rank}: {len(my_prompts)} prompts (indices {start(global_dp_rank)}-{start(global_dp_rank+1)-1})")
</syntaxhighlight>

=== Handling Empty Partitions ===
<syntaxhighlight lang="python">
# More workers than data
prompts = ["prompt_1", "prompt_2"]

global_dp_rank = 3  # Fourth worker
dp_size = 4

floor = len(prompts) // dp_size  # 0
remainder = len(prompts) % dp_size  # 2

def start(rank):
    return rank * floor + min(rank, remainder)

my_prompts = prompts[start(global_dp_rank):start(global_dp_rank + 1)]

if len(my_prompts) == 0:
    my_prompts = ["Placeholder"]
    print(f"Worker {global_dp_rank}: Using placeholder (no data assigned)")
else:
    print(f"Worker {global_dp_rank}: {len(my_prompts)} prompts")
</syntaxhighlight>

=== Complete Worker Function ===
<syntaxhighlight lang="python">
def partition_prompts_for_worker(prompts, rank, size):
    """Partition prompts for a specific worker.

    Args:
        prompts: Full list of prompts
        rank: Worker rank (0 to size-1)
        size: Total number of workers

    Returns:
        List of prompts for this worker (never empty)
    """
    floor = len(prompts) // size
    remainder = len(prompts) % size

    def start(r):
        return r * floor + min(r, remainder)

    # Extract partition
    worker_prompts = prompts[start(rank):start(rank + 1)]

    # Handle empty case
    if len(worker_prompts) == 0:
        worker_prompts = ["Placeholder"]

    return worker_prompts

# Usage in worker
my_prompts = partition_prompts_for_worker(
    prompts=all_prompts,
    rank=global_dp_rank,
    size=dp_size
)
</syntaxhighlight>

=== Verification Function ===
<syntaxhighlight lang="python">
def verify_partitioning(prompts, dp_size):
    """Verify that partitioning covers all data exactly once.

    Args:
        prompts: Full dataset
        dp_size: Number of workers

    Returns:
        bool: True if partitioning is valid
    """
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    # Check all partitions
    all_indices = []
    for rank in range(dp_size):
        partition_start = start(rank)
        partition_end = start(rank + 1)
        partition_indices = list(range(partition_start, partition_end))
        all_indices.extend(partition_indices)

    # Verify completeness and no duplicates
    expected_indices = list(range(len(prompts)))
    is_valid = sorted(all_indices) == expected_indices

    if is_valid:
        print(f"✓ Partitioning valid: {len(prompts)} items across {dp_size} workers")
    else:
        print(f"✗ Partitioning invalid!")

    return is_valid

# Test
prompts = ["p{}".format(i) for i in range(42)]
verify_partitioning(prompts, 5)
</syntaxhighlight>

=== Visualizing Partitions ===
<syntaxhighlight lang="python">
def visualize_partitions(total_items, dp_size):
    """Print partition sizes for all workers."""
    floor = total_items // dp_size
    remainder = total_items % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    print(f"Partitioning {total_items} items across {dp_size} workers:")
    print(f"  Base size: {floor}, Remainder: {remainder}")
    print()

    for rank in range(dp_size):
        partition_start = start(rank)
        partition_end = start(rank + 1)
        partition_size = partition_end - partition_start

        print(f"  Worker {rank}: indices [{partition_start:3d}, {partition_end:3d}) = {partition_size} items")

# Example
visualize_partitions(100, 7)
# Output:
# Partitioning 100 items across 7 workers:
#   Base size: 14, Remainder: 2
#
#   Worker 0: indices [  0,  15) = 15 items
#   Worker 1: indices [ 15,  30) = 15 items
#   Worker 2: indices [ 30,  44) = 14 items
#   Worker 3: indices [ 44,  58) = 14 items
#   Worker 4: indices [ 58,  72) = 14 items
#   Worker 5: indices [ 72,  86) = 14 items
#   Worker 6: indices [ 86, 100) = 14 items
</syntaxhighlight>

== Implementation Details ==

=== Start Index Formula ===
The key formula computes the start index for rank r:
<syntaxhighlight lang="python">
start(r) = r × floor + min(r, remainder)
</syntaxhighlight>

'''Intuition''':
* Each worker gets <code>floor</code> items: <code>r × floor</code>
* First <code>remainder</code> workers get 1 extra: <code>min(r, remainder)</code>
* For r < remainder: adds r extra items (one for each earlier worker plus this one)
* For r ≥ remainder: adds remainder extra items (all distributed to earlier workers)

=== Partition Size Formula ===
Size of partition for rank r:
<syntaxhighlight lang="python">
size(r) = start(r+1) - start(r)
        = floor + (1 if r < remainder else 0)
</syntaxhighlight>

=== Edge Cases ===
{| class="wikitable"
|-
! Case !! Behavior
|-
| dp_size = 1 || Single worker gets all data
|-
| len(prompts) = 0 || All workers get placeholder
|-
| dp_size > len(prompts) || First len(prompts) workers get 1 item each, rest get placeholder
|-
| len(prompts) % dp_size = 0 || All workers get exactly floor items
|}

== Performance Characteristics ==

=== Computational Efficiency ===
* '''Index Calculation''': O(1) per worker
* '''Slice Extraction''': O(k) where k is partition size
* '''Memory''': O(k) for partition copy
* '''No Communication''': Embarrassingly parallel

=== Load Balance ===
* '''Max Imbalance''': 1 item
* '''Workers with Extra Item''': First remainder workers
* '''Load Balance Factor''': ≥ floor/ceil(floor+1) ≥ 0.5 (for floor≥1)

== Best Practices ==

# '''Reusable Function''': Wrap in function for reuse across codebase
# '''Validation''': Verify partitioning in tests
# '''Documentation''': Document assumptions about data order
# '''Empty Handling''': Always handle empty partition case
# '''Index Bounds''': Verify start/end within valid range

== Related Pages ==
* [[implements::Principle:vllm-project_vllm_prompt_partitioning]] - Data partitioning principle
* [[related_to::vllm-project_vllm_LLM_generate_dp]] - Using partitioned data for inference
* [[related_to::vllm-project_vllm_result_aggregation]] - Aggregating results from partitions

== See Also ==
* examples/offline_inference/data_parallel.py - Reference implementation
* Python list slicing documentation
* Load balancing algorithms
