= Distributed Environment Setup =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vllm/config/parallel.py, examples/offline_inference/data_parallel.py, vllm/envs.py
|-
| Domains || Distributed Systems, Environment Configuration, Process Management
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Distributed Environment Setup principle defines how to configure the runtime environment for data parallel inference in vLLM. This principle guides the proper initialization of environment variables and process coordination mechanisms that enable multiple processes to work together as a distributed system.

== Description ==
vLLM uses environment variables as the primary mechanism for configuring data parallel workers. Each worker process must be configured with its identity (rank), the total size of the distributed system, and coordination endpoints. This approach enables:

* '''Process Identity''': Each worker knows its unique rank and role
* '''System Topology''': Workers understand the full distributed setup
* '''Communication Endpoints''': Workers can discover and connect to each other
* '''Isolation''': Different DP setups can coexist on the same system

The environment-based configuration separates concerns between the launcher (which spawns processes) and the workers (which perform inference), enabling flexible deployment across different execution environments.

== Environment Variables ==

=== Core Data Parallelism Variables ===
{| class="wikitable"
|-
! Variable !! Type !! Description !! Example
|-
| VLLM_DP_RANK || int || Global rank of this DP worker (0 to DP_SIZE-1) || "0"
|-
| VLLM_DP_RANK_LOCAL || int || Local rank of this DP worker within a node || "0"
|-
| VLLM_DP_SIZE || int || Total number of DP workers || "4"
|-
| VLLM_DP_MASTER_IP || str || IP address of the master coordinator || "127.0.0.1" or "10.0.0.1"
|-
| VLLM_DP_MASTER_PORT || int || Port for master coordinator communication || "29500"
|}

=== Additional Configuration ===
{| class="wikitable"
|-
! Variable !! Type !! Description !! Example
|-
| CUDA_VISIBLE_DEVICES || str || GPU devices visible to this process (auto-set) || "0,1"
|-
| VLLM_ENABLE_V1_MULTIPROCESSING || str || Enable V1 multiprocessing mode || "1"
|-
| VLLM_ALL2ALL_BACKEND || str || Backend for MoE all2all operations || "naive"
|}

== Setup Patterns ==

=== Single-Node Setup ===
For data parallel inference on a single machine:

<syntaxhighlight lang="python">
import os
from multiprocessing import Process

def setup_worker(rank, size):
    """Configure environment for a DP worker."""
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = "29500"

    # Worker-specific setup continues...

# Launch multiple workers
workers = []
for rank in range(4):  # 4 DP workers
    p = Process(target=setup_worker, args=(rank, 4))
    p.start()
    workers.append(p)
</syntaxhighlight>

=== Multi-Node Setup ===
For distributed inference across multiple nodes:

<syntaxhighlight lang="python">
import os

def setup_worker_multinode(
    global_rank,
    local_rank,
    dp_size,
    master_ip,
    master_port
):
    """Configure environment for multi-node DP worker."""
    os.environ["VLLM_DP_RANK"] = str(global_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

# Node 0 (IP: 10.0.0.1) - launches ranks 0-1
setup_worker_multinode(
    global_rank=0,
    local_rank=0,
    dp_size=4,
    master_ip="10.0.0.1",
    master_port=29500
)

# Node 1 (IP: 10.0.0.2) - launches ranks 2-3
setup_worker_multinode(
    global_rank=2,
    local_rank=0,
    dp_size=4,
    master_ip="10.0.0.1",  # Points to node 0
    master_port=29500
)
</syntaxhighlight>

=== Hybrid TP+DP Setup ===
Combining tensor parallelism and data parallelism:

<syntaxhighlight lang="python">
import os
from multiprocessing import Process

def setup_hybrid_worker(dp_rank, dp_size, tp_size):
    """Setup worker with both TP and DP."""
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = "29500"

    # Each DP worker will use tp_size GPUs
    # CUDA_VISIBLE_DEVICES set automatically by engine
    from vllm import LLM
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=tp_size
    )

# 8 GPUs total: 2 DP workers, each using 4 GPUs for TP
for dp_rank in range(2):
    p = Process(
        target=setup_hybrid_worker,
        args=(dp_rank, 2, 4)
    )
    p.start()
</syntaxhighlight>

== Validation and Coordination ==

=== Environment Validation ===
Workers should validate their environment on startup:

<syntaxhighlight lang="python">
def validate_dp_environment():
    """Validate required DP environment variables."""
    required_vars = [
        "VLLM_DP_RANK",
        "VLLM_DP_SIZE",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT"
    ]

    for var in required_vars:
        if var not in os.environ:
            raise ValueError(f"Required environment variable {var} not set")

    # Validate rank is within bounds
    rank = int(os.environ["VLLM_DP_RANK"])
    size = int(os.environ["VLLM_DP_SIZE"])
    if not (0 <= rank < size):
        raise ValueError(f"Invalid rank {rank} for size {size}")
</syntaxhighlight>

=== Process Group Initialization ===
Environment variables enable process group formation:

<syntaxhighlight lang="python">
import torch.distributed as dist

def init_dp_process_group():
    """Initialize distributed process group using env vars."""
    rank = int(os.environ["VLLM_DP_RANK"])
    size = int(os.environ["VLLM_DP_SIZE"])
    master_ip = os.environ["VLLM_DP_MASTER_IP"]
    master_port = os.environ["VLLM_DP_MASTER_PORT"]

    # Initialize process group
    dist.init_process_group(
        backend="gloo",  # or "nccl" for GPU
        init_method=f"tcp://{master_ip}:{master_port}",
        rank=rank,
        world_size=size
    )
</syntaxhighlight>

== Design Rationale ==

=== Environment-Based Configuration ===
Using environment variables provides several advantages:
* '''Simplicity''': Standard Unix mechanism understood by all process management tools
* '''Isolation''': Each process has independent environment
* '''Flexibility''': Works with various launchers (multiprocessing, Ray, Slurm, Kubernetes)
* '''Debugging''': Easy to inspect and override for testing

=== Rank-Based Identity ===
Each worker has a unique rank (0 to N-1):
* Enables deterministic work distribution
* Simplifies debugging and logging
* Supports process-to-GPU mapping
* Facilitates checkpoint sharding

=== Master-Based Coordination ===
A designated master endpoint (IP + port):
* Provides rendezvous point for all workers
* Enables barrier synchronization
* Supports distributed store for metadata
* Simplifies multi-node deployment

== Common Pitfalls ==

# '''Port Conflicts''': Ensure master port is available and not used by other processes
# '''IP Accessibility''': Master IP must be reachable from all nodes in multi-node setup
# '''Rank Overlap''': Each worker must have unique global rank
# '''Timing Issues''': All workers should start within timeout window
# '''Environment Pollution''': Clean environment before setting DP variables

== Related Pages ==
* [[implemented_by::Implementation:vllm-project_vllm_process_launcher]] - Process launching implementation
* [[implements::vllm-project_vllm_strategy_planning]] - Strategy planning principle
* [[related_to::vllm-project_vllm_ParallelConfig]] - Configuration that reads these variables
* [[related_to::vllm-project_vllm_LLM_distributed]] - LLM that uses this environment

== See Also ==
* examples/offline_inference/data_parallel.py - Reference implementation
* vllm/config/parallel.py - ParallelConfig environment variable handling
* vllm/envs.py - Environment variable definitions
