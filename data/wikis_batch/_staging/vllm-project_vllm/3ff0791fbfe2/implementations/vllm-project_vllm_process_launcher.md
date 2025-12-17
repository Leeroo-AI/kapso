= Process Launcher for Data Parallelism =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || examples/offline_inference/data_parallel.py
|-
| Domains || Process Management, Distributed Systems, System Programming
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Process Launcher implements the worker spawning logic for data parallel inference in vLLM. It uses Python's <code>multiprocessing.Process</code> to create independent worker processes, each configured with appropriate environment variables and assigned to specific data partitions.

== Description ==
The process launcher is responsible for:

* Calculating the distribution of DP workers across nodes
* Spawning worker processes with proper configuration
* Setting up environment variables for each worker
* Managing process lifecycle (start, monitor, terminate)
* Handling timeouts and error conditions

The implementation uses the <code>multiprocessing</code> module to achieve true parallelism, with each process running independently and having its own Python interpreter, memory space, and GPU assignment.

== Code Reference ==

=== Source Location ===
* '''File''': <code>/tmp/praxium_repo_583nq7ea/examples/offline_inference/data_parallel.py</code>
* '''Lines''': 206-268

=== Key Functions ===
<syntaxhighlight lang="python">
def main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    GPUs_per_dp_rank,
    # ... other parameters
):
    """Worker entry point - runs in each spawned process."""
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Create LLM and run inference
    llm = LLM(model=model, tensor_parallel_size=GPUs_per_dp_rank)
    outputs = llm.generate(prompts, sampling_params)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from multiprocessing import Process
</syntaxhighlight>

== I/O Contract ==

=== Input Parameters ===
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| dp_size || int || Total number of data parallel workers
|-
| node_size || int || Total number of nodes in the cluster
|-
| node_rank || int || Rank of the current node (0 to node_size-1)
|-
| dp_master_ip || str || IP address of the master coordinator
|-
| dp_master_port || int || Port for master coordinator
|-
| model || str || Model name or path
|-
| tp_size || int || Tensor parallel size (GPUs per DP worker)
|-
| timeout || int || Timeout in seconds for worker processes
|}

=== Output ===
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| exit_code || int || 0 for success, non-zero for failure
|}

=== Side Effects ===
* Spawns multiple child processes
* Sets environment variables in child processes
* Allocates GPU resources to workers
* Creates network connections between workers

== Usage Examples ==

=== Single-Node Launch ===
<syntaxhighlight lang="python">
from multiprocessing import Process
from vllm.utils.network_utils import get_open_port

# Configuration
dp_size = 4
tp_size = 2
node_size = 1
node_rank = 0

# For single node, use localhost
dp_master_ip = "127.0.0.1"
dp_master_port = get_open_port()

# Calculate workers per node
dp_per_node = dp_size // node_size  # 4 workers on this node

# Launch worker processes
procs = []
for local_dp_rank in range(dp_per_node):
    global_dp_rank = node_rank * dp_per_node + local_dp_rank

    proc = Process(
        target=main,
        args=(
            "meta-llama/Llama-3.1-8B-Instruct",  # model
            dp_size,                              # dp_size
            local_dp_rank,                        # local_dp_rank
            global_dp_rank,                       # global_dp_rank
            dp_master_ip,                         # dp_master_ip
            dp_master_port,                       # dp_master_port
            tp_size,                              # GPUs_per_dp_rank
            # ... other args
        ),
    )
    proc.start()
    procs.append(proc)

# Wait for all workers to complete
exit_code = 0
for proc in procs:
    proc.join(timeout=300)  # 5 minute timeout
    if proc.exitcode is None:
        print(f"Killing unresponsive process {proc.pid}")
        proc.kill()
        exit_code = 1
    elif proc.exitcode:
        exit_code = proc.exitcode

exit(exit_code)
</syntaxhighlight>

=== Multi-Node Launch ===
<syntaxhighlight lang="python">
from multiprocessing import Process

# Node configuration
dp_size = 8        # 8 total workers
node_size = 2      # 2 nodes
node_rank = 0      # This is node 0
tp_size = 2        # 2 GPUs per worker

# Master is on node 0
master_addr = "10.0.0.1"
master_port = 13345

# Calculate local workers
dp_per_node = dp_size // node_size  # 4 workers per node

# Launch workers for this node
procs = []
for local_dp_rank in range(dp_per_node):
    # Global rank calculation
    # Node 0: ranks 0-3
    # Node 1: ranks 4-7
    global_dp_rank = node_rank * dp_per_node + local_dp_rank

    proc = Process(
        target=main,
        args=(
            "ibm-research/PowerMoE-3b",
            dp_size,
            local_dp_rank,
            global_dp_rank,
            master_addr,
            master_port,
            tp_size,
            # ... other args
        ),
    )
    proc.start()
    procs.append(proc)

# Wait for completion
for proc in procs:
    proc.join()
</syntaxhighlight>

=== ROCm-Specific Launch ===
<syntaxhighlight lang="python">
from multiprocessing import Process, set_start_method
from vllm.platforms import current_platform

# ROCm requires 'spawn' method
if current_platform.is_rocm():
    set_start_method("spawn", force=True)

# Launch as usual
procs = []
for rank in range(dp_size):
    proc = Process(target=main, args=(...))
    proc.start()
    procs.append(proc)
</syntaxhighlight>

=== With Timeout Handling ===
<syntaxhighlight lang="python">
import sys
from multiprocessing import Process

procs = []
timeout = 600  # 10 minutes

# Launch workers
for rank in range(dp_size):
    proc = Process(target=main, args=(...))
    proc.start()
    procs.append(proc)

# Monitor with timeout
exit_code = 0
for proc in procs:
    proc.join(timeout=timeout)

    if proc.exitcode is None:
        # Process didn't finish in time
        print(f"ERROR: Process {proc.pid} timeout after {timeout}s")
        proc.kill()
        exit_code = 1
    elif proc.exitcode != 0:
        # Process exited with error
        print(f"ERROR: Process {proc.pid} failed with code {proc.exitcode}")
        exit_code = proc.exitcode

sys.exit(exit_code)
</syntaxhighlight>

== Implementation Details ==

=== Rank Assignment Strategy ===
The launcher uses a specific strategy to assign ranks:
<syntaxhighlight lang="python">
# For each node
for local_dp_rank in range(dp_per_node):
    global_dp_rank = node_rank * dp_per_node + local_dp_rank
</syntaxhighlight>

This ensures:
* Sequential global ranks across nodes
* Node 0 gets ranks [0, dp_per_node)
* Node 1 gets ranks [dp_per_node, 2*dp_per_node)
* Local ranks always start from 0 on each node

=== GPU Assignment ===
GPU assignment is handled automatically by the vLLM engine:
* Each DP worker gets <code>tensor_parallel_size</code> GPUs
* CUDA_VISIBLE_DEVICES is set automatically within the engine
* Workers don't interfere with each other's GPU allocations

=== Error Handling ===
The launcher implements robust error handling:
* Monitors all child processes
* Detects timeouts using <code>proc.join(timeout=...)</code>
* Kills unresponsive processes
* Propagates error exit codes to parent
* Returns non-zero exit code if any worker fails

=== Process Start Method ===
Different platforms require different start methods:
* '''fork''' (default on Linux): Fast but can have issues with CUDA
* '''spawn''' (required on ROCm): Slower but more reliable
* Set explicitly with <code>set_start_method("spawn", force=True)</code>

== Best Practices ==

# '''Timeout Configuration''': Set timeouts based on model size and data volume
# '''Error Propagation''': Always check exit codes and propagate failures
# '''Resource Cleanup''': Kill unresponsive processes to free resources
# '''Platform Detection''': Use appropriate start method for platform
# '''Rank Validation''': Validate rank assignments match expected topology

== Common Issues ==

=== Port Already in Use ===
<syntaxhighlight lang="python">
from vllm.utils.network_utils import get_open_port

# Don't hardcode ports
dp_master_port = get_open_port()  # Finds available port
</syntaxhighlight>

=== Process Hangs ===
* Set appropriate timeouts
* Check network connectivity between nodes
* Verify GPU availability
* Monitor for deadlocks in worker initialization

=== GPU Allocation Conflicts ===
* Ensure dp_size × tp_size ≤ total GPUs
* Don't manually set CUDA_VISIBLE_DEVICES (engine handles it)
* Check for other processes using GPUs

== Related Pages ==
* [[implements::Principle:vllm-project_vllm_dp_env_vars]] - Environment setup principle
* [[related_to::vllm-project_vllm_LLM_distributed]] - LLM initialization in workers
* [[related_to::vllm-project_vllm_prompt_partitioning]] - Data distribution to workers
* [[related_to::multiprocessing.Process]] - Python multiprocessing documentation

== See Also ==
* examples/offline_inference/data_parallel.py - Full implementation
* Python multiprocessing documentation
* vllm/utils/network_utils.py - Network utility functions
