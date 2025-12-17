= Parallel Inference Execution =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vllm/entrypoints/llm.py, examples/offline_inference/data_parallel.py
|-
| Domains || Distributed Inference, Parallel Computing, Model Execution
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Parallel Inference Execution principle defines how vLLM workers independently execute inference on their partitioned data in a data parallel setup. This principle ensures workers operate independently while maintaining coordination for system-wide synchronization when needed.

== Description ==
In data parallel inference, each worker executes the <code>generate()</code> method on its partition of prompts. The key aspects are:

* '''Independent Execution''': Workers process their data without inter-worker communication during generation
* '''Identical Configuration''': All workers use the same sampling parameters and model configuration
* '''Synchronous Completion''': Workers process their entire partition before exiting
* '''Result Isolation''': Each worker produces independent outputs
* '''Optional Coordination''': Workers may synchronize for KV cache management or termination detection

This principle enables true parallelism by eliminating dependencies between workers during the core inference phase.

== Execution Model ==

=== Worker Independence ===
Each worker operates as an independent inference engine:
* Loads its own copy of model weights
* Maintains its own KV cache
* Processes its partition without waiting for others
* Generates outputs at its own pace

This independence provides:
* '''Fault Tolerance''': One worker's failure doesn't affect others
* '''Flexibility''': Workers can have different sampling parameters
* '''Simplicity''': No complex inter-worker synchronization
* '''Performance''': No communication overhead during generation

=== Generate Method Execution ===
The core execution flow in each worker:

<syntaxhighlight lang="python">
# 1. Worker receives its partition
prompts = partition_for_this_worker

# 2. Configure sampling
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 3. Execute inference independently
outputs = llm.generate(prompts, sampling_params)

# 4. Process local results
for output in outputs:
    process_output(output)
</syntaxhighlight>

The <code>generate()</code> method blocks until all prompts in the partition are processed.

=== Synchronization Points ===
While execution is independent, some synchronization occurs:

# '''Initialization''': Workers synchronize during engine startup
# '''KV Cache Sizing''': Workers agree on cache size based on minimum available memory
# '''Termination Detection''': Workers may coordinate to detect when all have finished
# '''Error Handling''': Worker failures propagated to parent process

== Execution Patterns ==

=== Basic Parallel Execution ===
<syntaxhighlight lang="python">
import os
from vllm import LLM, SamplingParams

def worker_main(rank, size, all_prompts):
    """Worker entry point for parallel inference."""
    # Setup environment
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = "29500"

    # Partition data
    my_prompts = partition_data(all_prompts, rank, size)

    # Initialize engine
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    # Execute inference independently
    sampling_params = SamplingParams(max_tokens=100)
    outputs = llm.generate(my_prompts, sampling_params)

    # Return results
    return outputs
</syntaxhighlight>

=== With Different Sampling Per Worker ===
<syntaxhighlight lang="python">
def worker_main(rank, size, all_prompts):
    """Workers can use different sampling parameters."""
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(size)
    # ... env setup

    my_prompts = partition_data(all_prompts, rank, size)
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    # Different max_tokens per worker (for demonstration)
    max_tokens = [16, 20, 24, 28][rank % 4]
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )

    outputs = llm.generate(my_prompts, sampling_params)
    print(f"Worker {rank} generated with max_tokens={max_tokens}")
    return outputs
</syntaxhighlight>

=== With Progress Monitoring ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
import os

def worker_main_with_progress(rank, size, all_prompts):
    """Monitor progress during generation."""
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(size)
    # ... env setup

    my_prompts = partition_data(all_prompts, rank, size)

    print(f"Worker {rank}: Processing {len(my_prompts)} prompts")

    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
    sampling_params = SamplingParams(max_tokens=100)

    # Generate (blocking call)
    outputs = llm.generate(my_prompts, sampling_params)

    print(f"Worker {rank}: Completed {len(outputs)} outputs")

    # Print sample outputs
    for i, output in enumerate(outputs[:5]):  # First 5
        print(f"Worker {rank}, Output {i}: {output.outputs[0].text[:50]}...")

    return outputs
</syntaxhighlight>

=== Hybrid TP + DP Execution ===
<syntaxhighlight lang="python">
def worker_main_hybrid(rank, size, all_prompts, tp_size):
    """Worker with both tensor and data parallelism."""
    # Setup DP environment
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(size)
    # ... env setup

    # Partition data at DP level
    my_prompts = partition_data(all_prompts, rank, size)

    # Initialize with TP
    llm = LLM(
        model="meta-llama/Llama-3.1-70B-Instruct",
        tensor_parallel_size=tp_size  # Model split across tp_size GPUs
    )

    # Execute inference
    # TP workers within this DP worker coordinate automatically
    outputs = llm.generate(my_prompts, SamplingParams(max_tokens=100))

    print(f"DP Worker {rank} (using {tp_size} GPUs for TP): {len(outputs)} outputs")
    return outputs
</syntaxhighlight>

== Performance Characteristics ==

=== Throughput Scaling ===
Ideal data parallel scaling:
<syntaxhighlight lang="python">
# Single worker throughput: T tokens/sec
# With N workers: N × T tokens/sec (ideal)

# Actual throughput considering overhead:
# Throughput = N × T × efficiency
# Where efficiency ≈ 0.95-0.99 (communication + sync overhead)
</syntaxhighlight>

=== Latency Characteristics ===
Latency per request in DP setup:
* '''Individual Request''': Same as single worker
* '''Batch Processing''': Depends on partition size
* '''End-to-End''': Dominated by slowest worker

=== Resource Utilization ===
Each worker utilizes:
* '''GPU Compute''': ~100% during active generation
* '''GPU Memory''': Model weights + KV cache
* '''Network''': Minimal during generation (only for TP communication within worker)
* '''CPU''': Tokenization, scheduling, result processing

== Coordination Mechanisms ==

=== KV Cache Synchronization ===
Workers coordinate cache size at initialization:
<syntaxhighlight lang="python">
# Inside engine initialization
local_memory = compute_local_kv_cache_memory()

# Synchronize across DP workers (takes minimum)
global_memory = sync_across_dp_group(local_memory)

# All workers use same cache size
allocate_kv_cache(global_memory)
</syntaxhighlight>

=== Termination Detection ===
Workers may coordinate to detect completion:
<syntaxhighlight lang="python">
# Each worker tracks local completion
has_unfinished = check_local_requests()

# Aggregate across workers
global_has_unfinished = has_unfinished_dp(dp_group, has_unfinished)

if not global_has_unfinished:
    # All workers complete - can exit
    cleanup_and_exit()
</syntaxhighlight>

=== Error Propagation ===
Worker errors propagated to launcher:
<syntaxhighlight lang="python">
# In worker process
try:
    outputs = llm.generate(prompts, sampling_params)
except Exception as e:
    print(f"Worker {rank} failed: {e}")
    sys.exit(1)  # Non-zero exit code

# In launcher
proc.join()
if proc.exitcode != 0:
    print(f"Worker failed with code {proc.exitcode}")
    # Handle failure
</syntaxhighlight>

== Design Rationale ==

=== Independent Execution ===
Workers don't communicate during generation because:
* '''Performance''': Eliminates communication bottleneck
* '''Simplicity''': No complex coordination protocols
* '''Scalability''': Scales linearly with worker count
* '''Fault Tolerance''': Failures isolated to individual workers

=== Synchronous Completion ===
Workers complete all assigned work before exiting:
* '''Correctness''': Ensures all data processed
* '''Simplicity''': Clear completion semantics
* '''Debugging''': Easy to verify all work done

=== Result Isolation ===
Each worker produces independent results:
* '''Parallel I/O''': Workers can write results independently
* '''Fault Isolation''': Partial results preserved on failure
* '''Flexibility''': Different output formats per worker if needed

== Best Practices ==

# '''Balanced Partitioning''': Ensure workers have similar workload
# '''Error Handling''': Implement robust error handling in workers
# '''Progress Monitoring''': Log progress for long-running jobs
# '''Resource Limits''': Set appropriate timeouts
# '''Result Validation''': Verify output quality and completeness

== Common Issues ==

=== Load Imbalance ===
If workers have unequal workload:
* Some workers finish early, sit idle
* Overall time determined by slowest worker
* Solution: Better partitioning or dynamic scheduling

=== Memory Pressure ===
If workers run out of memory:
* Reduce <code>gpu_memory_utilization</code>
* Reduce <code>max_num_seqs</code>
* Use smaller partitions
* Enable swapping or offloading

=== Straggler Workers ===
If one worker is much slower:
* Check for hardware issues
* Verify partition sizes are balanced
* Monitor GPU utilization
* Check for resource contention

== Related Pages ==
* [[implemented_by::Implementation:vllm-project_vllm_generate_method]] - Generate method implementation
* [[implements::vllm-project_vllm_LLM_distributed]] - Distributed initialization
* [[related_to::vllm-project_vllm_prompt_partitioning]] - Data partitioning
* [[related_to::vllm-project_vllm_result_aggregation]] - Result aggregation
* [[related_to::SamplingParams]] - Sampling configuration

== See Also ==
* examples/offline_inference/data_parallel.py - Lines 175-200
* vllm/entrypoints/llm.py - LLM.generate() method
* Data parallel training patterns
