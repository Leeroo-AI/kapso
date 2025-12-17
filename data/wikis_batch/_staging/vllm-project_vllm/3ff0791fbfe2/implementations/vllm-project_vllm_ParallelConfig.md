= ParallelConfig =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vllm/config/parallel.py
|-
| Domains || Configuration Management, Distributed Systems
|-
| Last Updated || 2025-12-17
|}

== Overview ==
<code>ParallelConfig</code> is a configuration dataclass that encapsulates all settings related to distributed execution in vLLM. It manages parallelism dimensions (TP, DP, PP, CP, EP), backend selection, and runtime parameters for multi-GPU and multi-node inference.

== Description ==
The <code>ParallelConfig</code> class provides a centralized configuration system for vLLM's distributed execution capabilities. It handles:

* Configuration of all parallelism dimensions
* Validation of parallelism parameter combinations
* Automatic backend selection based on hardware and topology
* Process group initialization for distributed communication
* Hash computation for configuration validation across workers

The configuration is passed to engine initialization and used throughout the execution pipeline to coordinate distributed operations.

== Code Reference ==

=== Source Location ===
* '''File''': <code>/tmp/praxium_repo_583nq7ea/vllm/config/parallel.py</code>
* '''Lines''': 41-671

=== Class Definition ===
<syntaxhighlight lang="python">
@config
@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    data_parallel_size: int = 1
    data_parallel_size_local: int = 1
    data_parallel_rank: int = 0
    data_parallel_rank_local: int | None = None
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_rpc_port: int = 29550
    data_parallel_master_port: int = 29500
    data_parallel_backend: DataParallelBackend = "mp"
    # ... additional fields
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.config.parallel import ParallelConfig
</syntaxhighlight>

== I/O Contract ==

=== Input Parameters ===
{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| pipeline_parallel_size || int || 1 || Number of pipeline parallel groups
|-
| tensor_parallel_size || int || 1 || Number of tensor parallel groups
|-
| data_parallel_size || int || 1 || Number of data parallel groups
|-
| data_parallel_size_local || int || 1 || Number of local data parallel groups
|-
| data_parallel_rank || int || 0 || Rank of the data parallel group
|-
| data_parallel_master_ip || str || "127.0.0.1" || IP of the data parallel master
|-
| data_parallel_master_port || int || 29500 || Port of the data parallel master
|-
| data_parallel_backend || str || "mp" || Backend for data parallelism ("mp" or "ray")
|-
| enable_expert_parallel || bool || False || Use expert parallelism for MoE layers
|-
| expert_placement_strategy || str || "linear" || Expert placement strategy ("linear" or "round_robin")
|-
| distributed_executor_backend || str || None || Backend for distributed execution ("ray", "mp", "uni", "external_launcher")
|-
| nnodes || int || 1 || Number of nodes for multi-node inference
|-
| node_rank || int || 0 || Rank of current node
|-
| master_addr || str || "127.0.0.1" || Master address for multi-node communication
|-
| master_port || int || 29501 || Master port for multi-node communication
|}

=== Output ===
{| class="wikitable"
|-
! Property !! Type !! Description
|-
| world_size || int || Total number of processes (TP × PP × CP)
|-
| world_size_across_dp || int || Total processes including DP (TP × PP × CP × DP)
|-
| use_ray || bool || Whether Ray backend is being used
|-
| use_ubatching || bool || Whether microbatching is enabled
|}

=== Key Methods ===
{| class="wikitable"
|-
! Method !! Returns !! Description
|-
| get_next_dp_init_port() || int || Returns next available port for DP process group initialization
|-
| stateless_init_dp_group() || ProcessGroup || Initializes data parallel process group without side effects
|-
| compute_hash() || str || Computes configuration hash for validation across workers
|-
| has_unfinished_dp() || bool || Checks if any DP rank has unfinished work (static method)
|-
| sync_kv_cache_memory_size() || int || Synchronizes KV cache memory across DP ranks (static method)
|}

== Usage Examples ==

=== Basic Initialization ===
<syntaxhighlight lang="python">
from vllm.config.parallel import ParallelConfig

# Single-node TP configuration
parallel_config = ParallelConfig(
    tensor_parallel_size=4,
    data_parallel_size=1
)

# Multi-node TP + DP configuration
parallel_config = ParallelConfig(
    tensor_parallel_size=2,
    data_parallel_size=4,
    nnodes=2,
    node_rank=0,
    master_addr="10.0.0.1",
    master_port=29501
)
</syntaxhighlight>

=== MoE Configuration ===
<syntaxhighlight lang="python">
# Configuration for Mixture-of-Experts model
parallel_config = ParallelConfig(
    tensor_parallel_size=2,
    data_parallel_size=2,
    enable_expert_parallel=True,
    expert_placement_strategy="round_robin"
)
</syntaxhighlight>

=== Process Group Initialization ===
<syntaxhighlight lang="python">
# Initialize data parallel process group
parallel_config = ParallelConfig(
    data_parallel_size=4,
    data_parallel_rank=0,
    data_parallel_master_ip="127.0.0.1"
)

# Create process group for DP communication
dp_group = parallel_config.stateless_init_dp_group()
</syntaxhighlight>

=== Configuration Validation ===
<syntaxhighlight lang="python">
# Compute hash for configuration validation
config_hash = parallel_config.compute_hash()

# Use for validating consistency across workers
# All DP workers must have identical config hashes
</syntaxhighlight>

=== Environment Variable Integration ===
<syntaxhighlight lang="python">
import os

# ParallelConfig reads from environment variables when DP size not specified
os.environ["VLLM_DP_SIZE"] = "4"
os.environ["VLLM_DP_RANK"] = "0"
os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
os.environ["VLLM_DP_MASTER_PORT"] = "29500"

# Config automatically picks up env vars
parallel_config = ParallelConfig(
    tensor_parallel_size=2
    # data_parallel_size defaults to env var VLLM_DP_SIZE
)
</syntaxhighlight>

== Implementation Details ==

=== Automatic Backend Selection ===
The config automatically selects the distributed executor backend:
* Uses <code>mp</code> (multiprocessing) when all GPUs fit on current node
* Uses <code>ray</code> when Ray is available and configured
* Uses <code>uni</code> for single-process execution
* Validates multi-node setup requires appropriate backend

=== Validation Logic ===
The config includes comprehensive validation:
* Ensures <code>data_parallel_size_local ≤ data_parallel_size</code>
* Validates <code>data_parallel_rank</code> is within valid range
* Checks expert parallelism is only enabled with appropriate settings
* Verifies nnodes configuration with multi-node backends

=== Port Management ===
Manages port allocation for distributed communication:
* Maintains pool of available ports to avoid conflicts
* Provides <code>get_next_dp_init_port()</code> for sequential allocation
* Handles retries on port conflicts (EADDRINUSE)

== Related Pages ==
* [[implements::Principle:vllm-project_vllm_strategy_planning]] - Strategy planning principle
* [[related_to::vllm-project_vllm_dp_env_vars]] - Environment variable configuration
* [[related_to::vllm-project_vllm_LLM_distributed]] - LLM initialization using this config
* [[related_to::EngineArgs]] - Engine arguments that create ParallelConfig

== See Also ==
* vllm/config/parallel.py - Full source code
* vllm/distributed/parallel_state.py - Process group management
* vllm/engine/arg_utils.py - EngineArgs integration
