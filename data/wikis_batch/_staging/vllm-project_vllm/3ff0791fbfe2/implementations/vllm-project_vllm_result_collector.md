= Result Collector Implementation =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || examples/offline_inference/data_parallel.py
|-
| Domains || Process Management, Result Collection, Error Handling
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Result Collector Implementation provides the concrete mechanism for gathering outputs from data parallel workers in vLLM. This implementation uses process joining and exit code monitoring to ensure all workers complete successfully before the application terminates.

== Description ==
The result collector in vLLM's data parallel example implements a simple but robust pattern:

* Launch worker processes
* Monitor process completion with timeout
* Check exit codes for errors
* Kill unresponsive processes
* Propagate failures to parent

Workers produce results independently (printing to stdout or writing to storage), and the collector ensures the application doesn't exit until all workers finish or fail.

== Code Reference ==

=== Source Location ===
* '''File''': <code>/tmp/praxium_repo_583nq7ea/examples/offline_inference/data_parallel.py</code>
* '''Lines''': 258-268

=== Implementation ===
<syntaxhighlight lang="python">
# After launching all worker processes...

exit_code = 0
for proc in procs:
    proc.join(timeout=args.timeout)
    if proc.exitcode is None:
        print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
        proc.kill()
        exit_code = 1
    elif proc.exitcode:
        exit_code = proc.exitcode

exit(exit_code)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from multiprocessing import Process
</syntaxhighlight>

== I/O Contract ==

=== Input ===
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| procs || list[Process] || List of launched worker processes
|-
| timeout || int || Maximum seconds to wait for each process
|}

=== Output ===
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| exit_code || int || 0 for success, non-zero if any worker failed or timed out
|}

=== Side Effects ===
* Blocks until all workers complete or timeout
* Kills processes that exceed timeout
* Exits the application with appropriate code

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from multiprocessing import Process

# Launch workers
procs = []
for rank in range(dp_size):
    proc = Process(target=worker_main, args=(rank, ...))
    proc.start()
    procs.append(proc)

# Collect results with timeout
timeout = 300  # 5 minutes
exit_code = 0

for proc in procs:
    proc.join(timeout=timeout)

    if proc.exitcode is None:
        # Process didn't finish in time
        print(f"Killing unresponsive process {proc.pid}")
        proc.kill()
        exit_code = 1
    elif proc.exitcode != 0:
        # Process failed
        print(f"Process {proc.pid} failed with code {proc.exitcode}")
        exit_code = proc.exitcode

exit(exit_code)
</syntaxhighlight>

=== With Detailed Error Reporting ===
<syntaxhighlight lang="python">
from multiprocessing import Process
import sys

def collect_results_with_reporting(procs, timeout, dp_size):
    """Collect results with detailed error reporting."""
    exit_code = 0
    successful = []
    failed = []
    timeout_workers = []

    for rank, proc in enumerate(procs):
        proc.join(timeout=timeout)

        if proc.exitcode is None:
            # Timeout
            print(f"Worker {rank} (PID {proc.pid}) timed out after {timeout}s")
            proc.kill()
            timeout_workers.append(rank)
            exit_code = 1

        elif proc.exitcode == 0:
            # Success
            successful.append(rank)

        else:
            # Failure
            print(f"Worker {rank} (PID {proc.pid}) failed with code {proc.exitcode}")
            failed.append(rank)
            exit_code = proc.exitcode

    # Print summary
    print(f"\n=== Results Summary ===")
    print(f"Total workers: {dp_size}")
    print(f"Successful: {len(successful)} - {successful}")
    print(f"Failed: {len(failed)} - {failed}")
    print(f"Timeout: {len(timeout_workers)} - {timeout_workers}")

    return exit_code

# Usage
exit_code = collect_results_with_reporting(procs, 300, dp_size)
sys.exit(exit_code)
</syntaxhighlight>

=== With Result Queue ===
<syntaxhighlight lang="python">
from multiprocessing import Process, Queue

def collect_results_from_queue(procs, result_queue, timeout):
    """Collect results from queue and monitor processes."""

    # Wait for all processes to complete
    exit_code = 0
    for proc in procs:
        proc.join(timeout=timeout)

        if proc.exitcode is None:
            print(f"Killing process {proc.pid}")
            proc.kill()
            exit_code = 1
        elif proc.exitcode != 0:
            exit_code = proc.exitcode

    # Collect results from queue
    all_results = []
    while not result_queue.empty():
        try:
            worker_result = result_queue.get_nowait()
            all_results.extend(worker_result)
        except:
            break

    return all_results, exit_code

# Launch with queue
result_queue = Queue()
procs = []
for rank in range(dp_size):
    proc = Process(
        target=worker_with_queue,
        args=(rank, dp_size, prompts, result_queue)
    )
    proc.start()
    procs.append(proc)

# Collect
results, exit_code = collect_results_from_queue(procs, result_queue, 300)
print(f"Collected {len(results)} results")
exit(exit_code)
</syntaxhighlight>

=== With Incremental Collection ===
<syntaxhighlight lang="python">
from multiprocessing import Process
import time

def collect_results_incrementally(procs, timeout):
    """Monitor workers and collect results as they complete."""
    exit_code = 0
    completed_workers = []
    start_time = time.time()

    while len(completed_workers) < len(procs):
        # Check each process
        for rank, proc in enumerate(procs):
            if rank in completed_workers:
                continue

            # Check if process finished
            proc.join(timeout=0.1)  # Non-blocking check

            if proc.exitcode is not None:
                # Process completed
                if proc.exitcode == 0:
                    print(f"Worker {rank} completed successfully")
                else:
                    print(f"Worker {rank} failed with code {proc.exitcode}")
                    exit_code = proc.exitcode

                completed_workers.append(rank)

        # Check global timeout
        if time.time() - start_time > timeout:
            print(f"Global timeout reached ({timeout}s)")
            # Kill remaining workers
            for rank, proc in enumerate(procs):
                if rank not in completed_workers:
                    print(f"Killing worker {rank}")
                    proc.kill()
            exit_code = 1
            break

        time.sleep(0.5)  # Brief sleep to avoid busy waiting

    return exit_code

# Usage
exit_code = collect_results_incrementally(procs, 300)
exit(exit_code)
</syntaxhighlight>

=== With Retry Logic ===
<syntaxhighlight lang="python">
from multiprocessing import Process

def launch_with_retry(worker_func, rank, size, prompts, max_retries=3):
    """Launch worker with retry on failure."""
    for attempt in range(max_retries):
        proc = Process(target=worker_func, args=(rank, size, prompts))
        proc.start()
        proc.join(timeout=300)

        if proc.exitcode == 0:
            # Success
            return 0
        elif proc.exitcode is None:
            # Timeout
            print(f"Worker {rank} attempt {attempt+1} timed out")
            proc.kill()
        else:
            # Failure
            print(f"Worker {rank} attempt {attempt+1} failed with code {proc.exitcode}")

        if attempt < max_retries - 1:
            print(f"Retrying worker {rank}...")

    print(f"Worker {rank} failed after {max_retries} attempts")
    return 1

# Launch workers with retry
exit_code = 0
for rank in range(dp_size):
    result = launch_with_retry(worker_main, rank, dp_size, prompts)
    if result != 0:
        exit_code = result

exit(exit_code)
</syntaxhighlight>

=== With File-Based Result Collection ===
<syntaxhighlight lang="python">
from multiprocessing import Process
import os
import json

def collect_results_from_files(procs, dp_size, timeout):
    """Wait for workers and collect results from files."""

    # Wait for all workers
    exit_code = 0
    for rank, proc in enumerate(procs):
        proc.join(timeout=timeout)

        if proc.exitcode is None:
            print(f"Worker {rank} timeout")
            proc.kill()
            exit_code = 1
        elif proc.exitcode != 0:
            print(f"Worker {rank} failed")
            exit_code = proc.exitcode

    # If all succeeded, collect results from files
    if exit_code == 0:
        all_results = []
        for rank in range(dp_size):
            result_file = f"results_worker_{rank}.json"
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    worker_results = json.load(f)
                    all_results.extend(worker_results)

                # Clean up
                os.remove(result_file)
            else:
                print(f"Warning: Results file for worker {rank} not found")

        return all_results, exit_code
    else:
        return [], exit_code

# Usage
results, exit_code = collect_results_from_files(procs, dp_size, 300)
print(f"Collected {len(results)} results")
exit(exit_code)
</syntaxhighlight>

== Implementation Details ==

=== Process Joining ===
<code>proc.join(timeout=seconds)</code> behavior:
* Blocks for up to <code>timeout</code> seconds
* Returns when process exits or timeout expires
* <code>proc.exitcode</code> is None if still running
* <code>proc.exitcode</code> is 0 for success, non-zero for failure

=== Exit Code Handling ===
Exit code semantics:
* <code>0</code>: Worker completed successfully
* <code>None</code>: Worker still running (timeout)
* <code>non-zero</code>: Worker failed with error

The collector propagates the first non-zero exit code it encounters.

=== Process Termination ===
<code>proc.kill()</code> behavior:
* Sends SIGKILL to process (immediate termination)
* Use when process is unresponsive
* No cleanup code in worker will execute
* Process resources released by OS

=== Timeout Strategy ===
The timeout should account for:
* Model loading time
* Number of prompts per worker
* Generation length (max_tokens)
* GPU speed and availability
* Network latency (for multi-node)

Typical timeouts: 5-30 minutes for batch jobs.

== Performance Characteristics ==

=== Blocking Behavior ===
* <code>join()</code> blocks the collector thread
* Processes are checked sequentially
* Total wait time = sum of individual timeouts (worst case)
* Use shorter timeout + retry for responsiveness

=== Resource Management ===
* Killed processes release GPU memory immediately
* File descriptors and network connections cleaned up
* Shared memory may need explicit cleanup
* Zombie processes prevented by joining

== Best Practices ==

# '''Set Appropriate Timeout''': Based on workload characteristics
# '''Check All Exit Codes''': Don't stop at first failure
# '''Log Errors Clearly''': Include rank, PID, exit code
# '''Kill Unresponsive Processes''': Free resources promptly
# '''Propagate Failures''': Return non-zero exit code

== Common Issues ==

=== Zombie Processes ===
Processes not properly joined become zombies:
<syntaxhighlight lang="python">
# Always join processes
for proc in procs:
    proc.join(timeout=timeout)
    if proc.exitcode is None:
        proc.kill()
        proc.join()  # Join after kill to reap zombie
</syntaxhighlight>

=== Timeout Too Short ===
Workers killed prematurely:
* Increase timeout based on profiling
* Use adaptive timeout based on data size
* Log actual execution times to tune timeout

=== Exit Code Not Checked ===
Silent failures if exit codes ignored:
<syntaxhighlight lang="python">
# BAD: Ignores failures
for proc in procs:
    proc.join()

# GOOD: Checks exit codes
exit_code = 0
for proc in procs:
    proc.join()
    if proc.exitcode != 0:
        exit_code = proc.exitcode
</syntaxhighlight>

=== Results Not Collected ===
In this simple implementation, workers print results:
* For production, use Queue or files
* Ensure results persisted before process exits
* Verify all expected results received

== Related Pages ==
* [[implements::Principle:vllm-project_vllm_result_aggregation]] - Result aggregation principle
* [[related_to::vllm-project_vllm_process_launcher]] - Process launching
* [[related_to::vllm-project_vllm_LLM_generate_dp]] - Worker execution
* [[related_to::multiprocessing.Process]] - Process management

== See Also ==
* examples/offline_inference/data_parallel.py - Reference implementation
* Python multiprocessing.Process documentation
* Process monitoring and management patterns
