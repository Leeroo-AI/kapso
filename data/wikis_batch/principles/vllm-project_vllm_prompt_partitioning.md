= Data Partitioning for Parallel Inference =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || examples/offline_inference/data_parallel.py
|-
| Domains || Distributed Systems, Data Distribution, Load Balancing
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Data Partitioning principle defines how input data (prompts) are distributed across data parallel workers in vLLM. This principle ensures balanced workload distribution, prevents data duplication, and enables independent processing by each worker.

== Description ==
In data parallel inference, the total dataset is partitioned such that:

* '''Disjoint Partitions''': Each worker processes a unique subset of data (no overlap)
* '''Balanced Distribution''': Workers receive approximately equal amounts of work
* '''Deterministic Assignment''': Given rank and size, partition is reproducible
* '''Independent Processing''': Workers don't need to coordinate during processing

Data partitioning is the key mechanism that transforms a batch inference problem into multiple independent sub-problems that can be solved in parallel.

== Partitioning Strategy ==

=== Basic Partitioning Algorithm ===
The standard partitioning algorithm divides data evenly with remainder distribution:

<syntaxhighlight lang="python">
def partition_data(data, rank, size):
    """Partition data for a specific worker.

    Args:
        data: Full dataset (list of prompts)
        rank: Worker rank (0 to size-1)
        size: Total number of workers

    Returns:
        Partition for this worker
    """
    total_items = len(data)
    items_per_worker = total_items // size
    remainder = total_items % size

    # Start index calculation
    def start_index(r):
        return r * items_per_worker + min(r, remainder)

    # Extract partition
    start = start_index(rank)
    end = start_index(rank + 1)
    return data[start:end]
</syntaxhighlight>

=== Partition Size Calculation ===
Given total items N and workers W:
* '''Base size''': floor(N / W) items per worker
* '''Remainder''': N % W workers get one extra item
* '''Workers 0 to remainder-1''': Get floor(N/W) + 1 items
* '''Workers remainder to W-1''': Get floor(N/W) items

Example: 10 items, 4 workers
* Worker 0: items 0-2 (3 items)
* Worker 1: items 3-5 (3 items)
* Worker 2: items 6-7 (2 items)
* Worker 3: items 8-9 (2 items)

== Partitioning Patterns ==

=== Even Distribution ===
When dataset size is divisible by worker count:
<syntaxhighlight lang="python">
prompts = ["prompt_{}".format(i) for i in range(100)]
dp_size = 4
dp_rank = 0

# Each worker gets exactly 25 prompts
items_per_worker = len(prompts) // dp_size  # 25
start = dp_rank * items_per_worker
end = (dp_rank + 1) * items_per_worker
my_prompts = prompts[start:end]

print(f"Worker {dp_rank}: {len(my_prompts)} prompts")
# Worker 0: 25 prompts (0-24)
# Worker 1: 25 prompts (25-49)
# Worker 2: 25 prompts (50-74)
# Worker 3: 25 prompts (75-99)
</syntaxhighlight>

=== Uneven Distribution ===
When dataset size is not divisible:
<syntaxhighlight lang="python">
prompts = ["prompt_{}".format(i) for i in range(10)]
dp_size = 4
dp_rank = 1

floor = len(prompts) // dp_size  # 2
remainder = len(prompts) % dp_size  # 2

def start_index(rank):
    return rank * floor + min(rank, remainder)

start = start_index(dp_rank)
end = start_index(dp_rank + 1)
my_prompts = prompts[start:end]

print(f"Worker {dp_rank}: {len(my_prompts)} prompts")
# Worker 0: 3 prompts (0-2)
# Worker 1: 3 prompts (3-5)
# Worker 2: 2 prompts (6-7)
# Worker 3: 2 prompts (8-9)
</syntaxhighlight>

=== Handling Empty Partitions ===
When workers exceed data items:
<syntaxhighlight lang="python">
prompts = ["prompt_1", "prompt_2"]
dp_size = 4
dp_rank = 3

floor = len(prompts) // dp_size  # 0
remainder = len(prompts) % dp_size  # 2

def start_index(rank):
    return rank * floor + min(rank, remainder)

start = start_index(dp_rank)
end = start_index(dp_rank + 1)
my_prompts = prompts[start:end]

if len(my_prompts) == 0:
    # Use placeholder to prevent engine issues
    my_prompts = ["Placeholder"]

print(f"Worker {dp_rank}: {len(my_prompts)} prompts")
# Worker 0: 1 prompt (actual data)
# Worker 1: 1 prompt (actual data)
# Worker 2: 0 prompts -> 1 placeholder
# Worker 3: 0 prompts -> 1 placeholder
</syntaxhighlight>

=== Streaming Data Partitioning ===
For large datasets that don't fit in memory:
<syntaxhighlight lang="python">
def stream_partition(file_path, rank, size):
    """Stream only this worker's partition from file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    floor = total // size
    remainder = total % size

    def start_index(r):
        return r * floor + min(r, remainder)

    start = start_index(rank)
    end = start_index(rank + 1)

    return lines[start:end]

# Each worker reads only its partition
my_data = stream_partition("prompts.txt", dp_rank, dp_size)
</syntaxhighlight>

== Load Balancing Considerations ==

=== Workload Estimation ===
Different prompts may have different processing costs:
* '''Prompt Length''': Longer prompts take more time
* '''Generation Length''': Different max_tokens per prompt
* '''Complexity''': Some prompts may trigger more expensive operations

Simple partitioning assumes uniform cost. For better load balancing:

<syntaxhighlight lang="python">
def partition_by_length(prompts, rank, size):
    """Partition trying to balance total token count."""
    # Sort by length
    sorted_prompts = sorted(prompts, key=len)

    # Round-robin assignment for better balance
    my_prompts = sorted_prompts[rank::size]
    return my_prompts

# Example: Workers get mixed lengths
# Instead of: Worker 0 gets all short, Worker 3 gets all long
# Now: Each worker gets variety of lengths
</syntaxhighlight>

=== Dynamic Rebalancing ===
For online serving with dynamic load:
* Use external load balancer
* Route requests to least-loaded worker
* Monitor queue lengths and latencies
* Not applicable to offline batch inference

== Multi-Dimensional Partitioning ==

=== With Tensor Parallelism ===
When combining DP and TP:
<syntaxhighlight lang="python">
# Total: 8 GPUs
# Configuration: DP=4, TP=2
# Each DP worker uses 2 GPUs for TP

prompts = ["prompt_{}".format(i) for i in range(100)]

# Partition at DP level (4 partitions)
dp_rank = 0
dp_size = 4
items_per_worker = len(prompts) // dp_size

# Each DP worker gets 25 prompts
start = dp_rank * items_per_worker
end = (dp_rank + 1) * items_per_worker
my_prompts = prompts[start:end]

# TP workers within DP worker share the same prompts
# TP rank doesn't affect data partitioning
</syntaxhighlight>

=== Multi-Node Partitioning ===
Partition across nodes, then within nodes:
<syntaxhighlight lang="python">
prompts = ["prompt_{}".format(i) for i in range(100)]

# 2 nodes, 4 workers per node
node_rank = 0  # This is node 0
workers_per_node = 4
total_nodes = 2
total_workers = total_nodes * workers_per_node

# Each node gets half the data
items_per_node = len(prompts) // total_nodes
node_start = node_rank * items_per_node
node_end = (node_rank + 1) * items_per_node
node_prompts = prompts[node_start:node_end]

# Within node, partition to local workers
local_worker_rank = 0
items_per_local_worker = len(node_prompts) // workers_per_node
worker_start = local_worker_rank * items_per_local_worker
worker_end = (worker_start + 1) * items_per_local_worker
my_prompts = node_prompts[worker_start:worker_end]
</syntaxhighlight>

== Design Rationale ==

=== Disjoint Partitions ===
No overlap between worker partitions:
* '''Correctness''': Each item processed exactly once
* '''Efficiency''': No wasted computation
* '''Simplicity''': No deduplication needed

=== Rank-Based Assignment ===
Deterministic partitioning based on rank:
* '''Reproducibility''': Same rank always gets same data
* '''Debugging''': Easy to identify which worker processed what
* '''Fault Recovery''': Can restart failed worker with same partition

=== Remainder Distribution ===
First K workers get extra item (where K = N % W):
* '''Balance''': Minimizes difference between worker loads
* '''Simplicity''': Simple formula for start/end indices
* '''Fairness''': Distributes remainder evenly

== Best Practices ==

# '''Validate Partitions''': Verify all data covered and no overlap
# '''Handle Empty Case''': Provide placeholder when partition is empty
# '''Consider Data Distribution''': Sort or shuffle for better balance if needed
# '''Document Assumptions''': Clarify whether data is pre-shuffled
# '''Test Edge Cases''': Workers > data items, data items < workers, etc.

== Related Pages ==
* [[implemented_by::Implementation:vllm-project_vllm_data_partition_impl]] - Implementation code
* [[related_to::vllm-project_vllm_process_launcher]] - Process creation for workers
* [[related_to::vllm-project_vllm_LLM_generate_dp]] - Inference on partitioned data
* [[related_to::vllm-project_vllm_result_aggregation]] - Collecting results from partitions

== See Also ==
* examples/offline_inference/data_parallel.py - Lines 143-165
* Distributed data loading patterns
* Load balancing strategies
