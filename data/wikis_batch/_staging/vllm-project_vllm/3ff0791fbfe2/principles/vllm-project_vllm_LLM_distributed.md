= Distributed Engine Initialization =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vllm/entrypoints/llm.py, vllm/v1/engine/llm_engine.py, examples/offline_inference/data_parallel.py
|-
| Domains || Distributed Systems, Model Loading, Engine Architecture
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Distributed Engine Initialization principle defines how vLLM's LLM engine is instantiated in a data parallel context. This principle ensures each worker properly initializes its model replica, establishes distributed communication channels, and coordinates with other workers to form a cohesive distributed inference system.

== Description ==
In data parallel inference, each worker runs an independent instance of the LLM engine. However, these instances must coordinate to:

* '''Model Consistency''': All workers load identical model weights and configurations
* '''Process Group Formation''': Workers establish communication channels for synchronization
* '''Configuration Validation''': Verify all workers have compatible settings
* '''Resource Allocation''': Each worker claims exclusive access to its assigned GPUs
* '''Initialization Ordering''': Coordinate startup to avoid race conditions

The initialization process is environment-aware, reading configuration from environment variables set by the process launcher and automatically configuring distributed communication based on the detected topology.

== Initialization Phases ==

=== Phase 1: Environment Detection ===
The engine reads environment variables to understand its role:
* <code>VLLM_DP_RANK</code>: This worker's position in the distributed system
* <code>VLLM_DP_SIZE</code>: Total number of workers
* <code>VLLM_DP_MASTER_IP</code> / <code>VLLM_DP_MASTER_PORT</code>: Coordination endpoint

These variables inform the engine that it's part of a distributed setup and trigger data parallel initialization paths.

=== Phase 2: Configuration Creation ===
The engine creates configuration objects that define its behavior:
* '''ParallelConfig''': Parallelism dimensions (TP, DP, PP)
* '''ModelConfig''': Model architecture and weights
* '''CacheConfig''': KV cache allocation strategy
* '''SchedulerConfig''': Request batching and scheduling

The configuration hash is computed and will be validated across workers to ensure consistency.

=== Phase 3: Process Group Initialization ===
Workers establish distributed communication:
* Connect to the master coordinator
* Exchange rank information
* Form process groups for collective operations
* Set up communication backends (NCCL, Gloo)

This phase includes handshake protocols and timeout handling to ensure all workers successfully join the group.

=== Phase 4: Model Loading ===
Each worker loads the model independently:
* Download or access model weights from storage
* Load weights into GPU memory
* Apply tensor parallelism if configured
* Verify model consistency across workers

Model loading can be parallelized or sequential depending on configuration and available resources.

=== Phase 5: Resource Allocation ===
Workers allocate GPU resources:
* '''KV Cache Memory''': Synchronize cache sizes across workers
* '''Compute Resources''': Allocate scratch space for operations
* '''Communication Buffers''': Pre-allocate buffers for collectives

The smallest available memory across all workers determines the KV cache size to ensure consistent capacity.

== Configuration Parameters ==

Key parameters that affect distributed initialization:

{| class="wikitable"
|-
! Parameter !! Type !! Purpose
|-
| model || str || Model name or path (must be identical across workers)
|-
| tensor_parallel_size || int || GPUs per worker for tensor parallelism
|-
| dtype || str || Data type for weights and activations
|-
| gpu_memory_utilization || float || Fraction of GPU memory to use
|-
| trust_remote_code || bool || Allow executing custom model code
|-
| enforce_eager || bool || Disable CUDA graphs for debugging
|-
| max_model_len || int || Maximum sequence length
|-
| max_num_seqs || int || Maximum concurrent sequences
|}

== Initialization Patterns ==

=== Basic Initialization ===
<syntaxhighlight lang="python">
import os
from vllm import LLM

# Environment already configured by launcher
# VLLM_DP_RANK, VLLM_DP_SIZE, etc.

# Simple initialization - engine auto-detects DP mode
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2
)
# Engine automatically:
# - Reads DP env vars
# - Joins DP process group
# - Synchronizes with other workers
</syntaxhighlight>

=== With Custom Configuration ===
<syntaxhighlight lang="python">
from vllm import LLM

llm = LLM(
    model="ibm-research/PowerMoE-3b",
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    gpu_memory_utilization=0.9,
    max_num_seqs=64,
    max_model_len=2048,
    trust_remote_code=True,
    enforce_eager=False
)
</syntaxhighlight>

=== Verification After Init ===
<syntaxhighlight lang="python">
from vllm import LLM
import os

# Initialize engine
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Verify DP configuration
dp_rank = int(os.environ.get("VLLM_DP_RANK", 0))
dp_size = int(os.environ.get("VLLM_DP_SIZE", 1))

print(f"Worker {dp_rank}/{dp_size} initialized successfully")
print(f"Model loaded: {llm.llm_engine.model_config.model}")
</syntaxhighlight>

== Synchronization Points ==

The initialization process includes several synchronization points where workers coordinate:

=== Config Hash Validation ===
All workers compute and exchange configuration hashes to ensure compatibility:
<syntaxhighlight lang="python">
# Inside engine initialization
config_hash = parallel_config.compute_hash()
# All workers must have identical hashes
# Mismatch causes initialization failure
</syntaxhighlight>

=== KV Cache Size Synchronization ===
Workers negotiate the KV cache size based on available memory:
<syntaxhighlight lang="python">
# Each worker computes its available memory
local_kv_cache_memory = compute_available_memory()

# Synchronize across DP group (take minimum)
global_kv_cache_memory = ParallelConfig.sync_kv_cache_memory_size(
    dp_group,
    local_kv_cache_memory
)
# All workers use the minimum available memory
</syntaxhighlight>

=== Handshake Protocol ===
Workers perform handshake to confirm successful initialization:
* Rank 0 (master) waits for all workers to connect
* Each worker reports successful model loading
* Master broadcasts "ready" signal
* All workers proceed to inference phase

== Error Handling ==

=== Configuration Mismatch ===
If workers have incompatible configurations:
<syntaxhighlight lang="python">
# Example error scenario
# Worker 0: tensor_parallel_size=2
# Worker 1: tensor_parallel_size=4
# Result: Configuration hash mismatch
# Action: Initialization fails with clear error message
</syntaxhighlight>

=== Timeout Handling ===
If workers fail to connect within timeout:
<syntaxhighlight lang="python">
# Process group initialization with retry
try:
    dp_group = parallel_config.stateless_init_dp_group()
except DistNetworkError as e:
    if "EADDRINUSE" in str(e):
        # Port conflict - retry with new port
        retry_with_new_port()
    else:
        raise
</syntaxhighlight>

=== Resource Allocation Failure ===
If GPU memory is insufficient:
* Engine computes required memory
* Compares with available memory
* Raises OOM error with diagnostic information
* Suggests configuration adjustments

== Design Rationale ==

=== Independent Engine Instances ===
Each worker runs a complete, independent engine rather than sharing state:
* '''Fault Isolation''': Worker failures don't affect others
* '''Simplicity''': No complex state sharing or locking
* '''Flexibility''': Workers can run different sampling parameters
* '''Debugging''': Each worker can be inspected independently

=== Environment-Based Configuration ===
Using environment variables for DP setup:
* '''Transparency''': Clear separation between launcher and worker logic
* '''Compatibility''': Works with any process management system
* '''Debugging''': Easy to override for testing
* '''No Code Changes''': Same code works in DP and non-DP modes

=== Lazy Process Group Creation ===
Process groups created on-demand rather than upfront:
* '''Resource Efficiency''': Only create what's needed
* '''Robustness''': Reduce initialization complexity
* '''Flexibility''': Support different communication patterns

== Best Practices ==

# '''Identical Configurations''': Ensure all workers use same model, dtype, and settings
# '''Adequate Memory''': Plan for KV cache based on smallest worker's available memory
# '''Network Connectivity''': Verify all workers can reach master coordinator
# '''Error Logging''': Enable detailed logging for initialization debugging
# '''Timeout Tuning''': Adjust timeouts based on model size and network latency

== Related Pages ==
* [[implemented_by::Implementation:vllm-project_vllm_LLM_class]] - LLM class implementation
* [[implements::vllm-project_vllm_dp_env_vars]] - Environment setup principle
* [[related_to::vllm-project_vllm_ParallelConfig]] - Configuration object
* [[related_to::vllm-project_vllm_LLM_generate_dp]] - Inference execution after init

== See Also ==
* vllm/entrypoints/llm.py - LLM class implementation
* vllm/v1/engine/llm_engine.py - Engine core implementation
* examples/offline_inference/data_parallel.py - Usage example
