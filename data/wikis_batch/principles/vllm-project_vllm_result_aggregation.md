= Distributed Result Aggregation =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || examples/offline_inference/data_parallel.py
|-
| Domains || Distributed Systems, Data Processing, Result Collection
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Distributed Result Aggregation principle defines how outputs from multiple data parallel workers are collected, organized, and presented. This principle addresses the challenge of reconstructing a coherent result set from independently executed worker processes.

== Description ==
After workers complete their independent inference execution, the system must:

* '''Collect Results''': Gather outputs from all worker processes
* '''Maintain Ordering''': Preserve the original input order across workers
* '''Handle Errors''': Detect and report worker failures
* '''Aggregate Metadata''': Combine statistics and metrics from workers
* '''Present Unified View''': Provide results as if from single execution

The aggregation strategy depends on the deployment pattern:
* '''Offline Batch''': Results collected after all workers complete
* '''Online Streaming''': Results streamed as they become available
* '''Application-Level''': Custom aggregation logic in application code

== Aggregation Strategies ==

=== Post-Execution Collection ===
The simplest strategy: collect all results after workers finish:

<syntaxhighlight lang="python">
from multiprocessing import Process

def launch_workers(prompts, dp_size):
    """Launch workers and collect results."""
    # Store results from each worker
    results = [None] * dp_size

    # Launch workers
    procs = []
    for rank in range(dp_size):
        proc = Process(
            target=worker_main,
            args=(rank, dp_size, prompts, results)
        )
        proc.start()
        procs.append(proc)

    # Wait for all workers
    for proc in procs:
        proc.join()

    # Aggregate results
    all_outputs = []
    for worker_results in results:
        if worker_results:
            all_outputs.extend(worker_results)

    return all_outputs
</syntaxhighlight>

=== File-Based Aggregation ===
Workers write results to files, launcher reads them:

<syntaxhighlight lang="python">
import json
import os

def worker_main(rank, size, prompts):
    """Worker writes results to file."""
    # ... inference execution ...

    # Write results to worker-specific file
    output_file = f"results_worker_{rank}.json"
    with open(output_file, 'w') as f:
        results = [
            {
                "prompt": output.prompt,
                "generated": output.outputs[0].text,
                "rank": rank
            }
            for output in outputs
        ]
        json.dump(results, f)

def aggregate_results(dp_size):
    """Aggregate results from all worker files."""
    all_results = []

    for rank in range(dp_size):
        output_file = f"results_worker_{rank}.json"
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                worker_results = json.load(f)
                all_results.extend(worker_results)

    # Sort by original order if needed
    return all_results
</syntaxhighlight>

=== Shared Memory Aggregation ===
Workers write to shared memory structure:

<syntaxhighlight lang="python">
from multiprocessing import Manager

def launch_with_shared_memory(prompts, dp_size):
    """Use shared memory for result collection."""
    manager = Manager()
    shared_results = manager.list([None] * dp_size)

    procs = []
    for rank in range(dp_size):
        proc = Process(
            target=worker_with_shared_mem,
            args=(rank, dp_size, prompts, shared_results)
        )
        proc.start()
        procs.append(proc)

    # Wait for completion
    for proc in procs:
        proc.join()

    # Aggregate
    all_outputs = []
    for worker_results in shared_results:
        if worker_results:
            all_outputs.extend(worker_results)

    return all_outputs

def worker_with_shared_mem(rank, size, prompts, shared_results):
    """Worker stores results in shared memory."""
    # ... inference execution ...

    # Store in shared list at this worker's position
    shared_results[rank] = outputs
</syntaxhighlight>

== Result Ordering ==

=== Preserving Input Order ===
Maintaining original prompt order across workers:

<syntaxhighlight lang="python">
def aggregate_with_ordering(worker_results, dp_size):
    """Aggregate results while preserving original order.

    Args:
        worker_results: List of results from each worker
        dp_size: Number of workers

    Returns:
        Results in original prompt order
    """
    # Each worker processed a contiguous partition
    # Worker 0: prompts [0, k)
    # Worker 1: prompts [k, 2k)
    # etc.

    # Concatenate in worker rank order
    ordered_results = []
    for rank in range(dp_size):
        if worker_results[rank]:
            ordered_results.extend(worker_results[rank])

    return ordered_results
</syntaxhighlight>

=== Index-Based Reordering ===
When workers process data out of order:

<syntaxhighlight lang="python">
def aggregate_with_indices(worker_results):
    """Aggregate using explicit indices."""
    # Each result includes its original index
    all_results = []
    for worker_result in worker_results:
        all_results.extend(worker_result)

    # Sort by original index
    all_results.sort(key=lambda x: x.original_index)

    return all_results

# Worker includes index in output
def worker_with_indices(rank, size, prompts):
    # ... inference ...

    results = [
        {
            "original_index": start_index + i,
            "prompt": output.prompt,
            "generated": output.outputs[0].text
        }
        for i, output in enumerate(outputs)
    ]
    return results
</syntaxhighlight>

== Error Handling ==

=== Detecting Worker Failures ===
<syntaxhighlight lang="python">
from multiprocessing import Process

def launch_with_error_detection(prompts, dp_size):
    """Launch workers and detect failures."""
    procs = []
    results = [None] * dp_size

    # Launch workers
    for rank in range(dp_size):
        proc = Process(
            target=worker_main,
            args=(rank, dp_size, prompts, results)
        )
        proc.start()
        procs.append(proc)

    # Wait and check exit codes
    failed_workers = []
    for rank, proc in enumerate(procs):
        proc.join()

        if proc.exitcode != 0:
            failed_workers.append(rank)
            print(f"ERROR: Worker {rank} failed with code {proc.exitcode}")

    # Check if aggregation is possible
    if failed_workers:
        raise RuntimeError(
            f"Workers {failed_workers} failed. "
            f"Cannot complete aggregation."
        )

    # Aggregate successful results
    return aggregate_results(results)
</syntaxhighlight>

=== Partial Result Collection ===
Collecting results even if some workers fail:

<syntaxhighlight lang="python">
def aggregate_partial_results(worker_results, dp_size):
    """Aggregate available results, skip failed workers.

    Args:
        worker_results: List of results (None for failed workers)
        dp_size: Total number of workers

    Returns:
        Partial results with metadata about failures
    """
    successful_workers = []
    failed_workers = []
    all_outputs = []

    for rank in range(dp_size):
        if worker_results[rank] is not None:
            successful_workers.append(rank)
            all_outputs.extend(worker_results[rank])
        else:
            failed_workers.append(rank)

    print(f"Successful workers: {successful_workers}")
    print(f"Failed workers: {failed_workers}")
    print(f"Collected {len(all_outputs)} results from {len(successful_workers)} workers")

    return {
        "outputs": all_outputs,
        "successful_workers": successful_workers,
        "failed_workers": failed_workers,
        "completion_rate": len(successful_workers) / dp_size
    }
</syntaxhighlight>

== Metadata Aggregation ==

=== Combining Statistics ===
<syntaxhighlight lang="python">
def aggregate_statistics(worker_stats):
    """Combine statistics from all workers.

    Args:
        worker_stats: List of stat dicts from each worker

    Returns:
        Aggregated statistics
    """
    total_prompts = sum(s["num_prompts"] for s in worker_stats)
    total_tokens = sum(s["total_tokens"] for s in worker_stats)
    total_time = max(s["execution_time"] for s in worker_stats)

    avg_throughput = total_tokens / total_time if total_time > 0 else 0

    return {
        "total_prompts": total_prompts,
        "total_tokens_generated": total_tokens,
        "total_time_seconds": total_time,
        "average_throughput_tokens_per_sec": avg_throughput,
        "num_workers": len(worker_stats)
    }

# Worker tracks statistics
def worker_with_stats(rank, size, prompts):
    import time
    start_time = time.time()

    # ... inference ...

    execution_time = time.time() - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    stats = {
        "num_prompts": len(outputs),
        "total_tokens": total_tokens,
        "execution_time": execution_time
    }

    return outputs, stats
</syntaxhighlight>

=== Performance Metrics ===
<syntaxhighlight lang="python">
def aggregate_performance_metrics(worker_metrics):
    """Aggregate performance metrics across workers.

    Args:
        worker_metrics: Performance data from each worker

    Returns:
        System-wide performance metrics
    """
    metrics = {
        "workers": len(worker_metrics),
        "total_throughput": sum(m["throughput"] for m in worker_metrics),
        "avg_latency": sum(m["latency"] for m in worker_metrics) / len(worker_metrics),
        "min_latency": min(m["latency"] for m in worker_metrics),
        "max_latency": max(m["latency"] for m in worker_metrics),
        "load_balance_ratio": min(m["throughput"] for m in worker_metrics) /
                            max(m["throughput"] for m in worker_metrics)
    }

    return metrics
</syntaxhighlight>

== Practical Patterns ==

=== Simple In-Process Aggregation ===
The most common pattern for offline inference:

<syntaxhighlight lang="python">
from multiprocessing import Process

def run_data_parallel_inference(prompts, dp_size):
    """Run DP inference and return aggregated results."""

    # Launch workers (each returns results via return value)
    # Note: This requires using Queue or Manager for IPC

    from multiprocessing import Manager

    manager = Manager()
    results_queue = manager.Queue()

    procs = []
    for rank in range(dp_size):
        proc = Process(
            target=worker_with_queue,
            args=(rank, dp_size, prompts, results_queue)
        )
        proc.start()
        procs.append(proc)

    # Wait for all workers
    for proc in procs:
        proc.join()

    # Collect results from queue
    all_results = []
    while not results_queue.empty():
        worker_result = results_queue.get()
        all_results.extend(worker_result)

    return all_results

def worker_with_queue(rank, size, prompts, results_queue):
    """Worker puts results in queue."""
    # ... inference ...

    results_queue.put(outputs)
</syntaxhighlight>

=== Application-Level Aggregation ===
For production systems with custom requirements:

<syntaxhighlight lang="python">
def application_aggregation(prompts, dp_size):
    """Application-level result aggregation."""

    # Workers write to application-specific storage
    # (e.g., database, object store, message queue)

    for rank in range(dp_size):
        worker_with_db_output(rank, size, prompts)

    # Application aggregates from storage
    results = fetch_from_database(prompts)

    return results
</syntaxhighlight>

== Design Rationale ==

=== Independent Worker Outputs ===
Workers produce independent outputs because:
* '''Simplicity''': No coordination during generation
* '''Fault Tolerance''': Partial results can be saved
* '''Flexibility''': Different aggregation strategies possible
* '''Performance''': No communication overhead

=== Order Preservation ===
Maintaining input order is important for:
* '''Correctness''': Match outputs to inputs
* '''Reproducibility''': Consistent results across runs
* '''Debugging''': Easy to trace prompts to outputs
* '''User Experience''': Predictable output structure

=== Post-Execution Collection ===
Collecting results after execution completes:
* '''Simplicity''': Straightforward implementation
* '''Reliability''': All-or-nothing semantics
* '''Synchronization''': Clear completion point
* '''Suitability''': Ideal for offline batch processing

== Best Practices ==

# '''Preserve Order''': Maintain original prompt order in results
# '''Handle Failures''': Implement robust error detection
# '''Track Metadata''': Collect statistics for monitoring
# '''Verify Completeness''': Ensure all expected results received
# '''Clean Up Resources''': Remove temporary files/shared memory

== Common Issues ==

=== Lost Results ===
* Worker crash before writing results
* File I/O errors
* Shared memory corruption
* Solution: Implement retries and checksums

=== Out-of-Order Results ===
* Workers finish at different times
* No explicit ordering maintained
* Solution: Include indices or use deterministic partitioning

=== Memory Pressure ===
* Large result sets in memory
* Shared memory exhaustion
* Solution: Stream results to disk, process incrementally

== Related Pages ==
* [[implemented_by::Implementation:vllm-project_vllm_result_collector]] - Result collection implementation
* [[related_to::vllm-project_vllm_LLM_generate_dp]] - Parallel execution
* [[related_to::vllm-project_vllm_prompt_partitioning]] - Data partitioning
* [[related_to::vllm-project_vllm_process_launcher]] - Worker process management

== See Also ==
* examples/offline_inference/data_parallel.py - Lines 189-200
* Python multiprocessing documentation
* Distributed result aggregation patterns
