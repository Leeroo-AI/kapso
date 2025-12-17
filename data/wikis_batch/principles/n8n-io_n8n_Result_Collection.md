{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Process_Management]], [[domain::IPC]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Collecting execution results from isolated subprocesses via inter-process communication while handling timeouts, signals, and error conditions.

=== Description ===

Result collection manages the communication between parent and child processes to retrieve execution results, output logs, and error information. The process involves:

1. **Pipe-Based IPC**: Child process writes serialized results to a pipe
2. **Concurrent Reading**: Background thread reads pipe while waiting for process
3. **Timeout Enforcement**: Sends termination signals if execution exceeds time limit
4. **Exit Code Interpretation**: Distinguishes normal completion from crashes
5. **Signal Handling**: Interprets SIGTERM, SIGKILL, SIGSEGV to determine failure mode

The collection process must handle multiple asynchronous events:
* Process termination (normal or abnormal)
* Timeout expiration requiring forced termination
* Pipe data arrival (potentially large, requiring chunked reads)
* Various failure modes (crashes, hangs, resource exhaustion)

This coordination ensures that results are reliably retrieved when execution succeeds, and appropriate error information is captured when execution fails, all while enforcing strict time bounds on execution.

=== Usage ===

Apply this principle when designing systems that execute code in subprocesses where:
* Results must be retrieved across process boundaries
* Execution has strict time limits requiring timeout enforcement
* System must distinguish between success, timeout, and crash
* Large result sets require streaming rather than single read
* Parent process must remain responsive during child execution
* Multiple concurrent subprocesses require management
* Graceful vs forceful termination has different semantics

== Theoretical Basis ==

Result collection implements **concurrent IPC with timeout-controlled process management**:

**Architecture:**

```
Main Thread:                 Background Thread:           Subprocess:
  start_process()
  start_reader_thread()  →   read_loop()
  wait(timeout)                 ├─ pipe.read()         → write(result)
    ├─ timeout expired?          ├─ deserialize           write(output)
    │   ├─ send SIGTERM          └─ accumulate           exit(0)
    │   ├─ wait(grace)
    │   └─ send SIGKILL
    └─ process.wait()
  collect_results()
  return result
```

**Pipe Communication:**

```python
# Pipe creation (unidirectional)
parent_conn, child_conn = multiprocessing.Pipe(duplex=False)

# Child writes results
child_conn.send({
    'result': processed_data,
    'output': captured_logs,
    'metadata': execution_info
})

# Parent reads results (in background thread)
if parent_conn.poll(timeout=0.1):  # Non-blocking check
    data = parent_conn.recv()       # Blocking read
```

**Timeout Enforcement:**

```python
def wait_with_timeout(process, timeout):
    try:
        exit_code = process.wait(timeout=timeout)
        return exit_code, 'completed'
    except TimeoutExpired:
        # Graceful termination
        process.terminate()  # SIGTERM
        try:
            process.wait(timeout=grace_period)
            return None, 'timeout_graceful'
        except TimeoutExpired:
            # Forceful termination
            process.kill()  # SIGKILL
            process.wait()
            return None, 'timeout_killed'
```

**Signal Interpretation:**

```python
def interpret_exit(process):
    exit_code = process.returncode

    if exit_code == 0:
        return 'success'
    elif exit_code < 0:
        # Negative exit codes indicate signals
        signal_num = -exit_code
        if signal_num == signal.SIGTERM:
            return 'terminated'
        elif signal_num == signal.SIGKILL:
            return 'killed'
        elif signal_num == signal.SIGSEGV:
            return 'segfault'
    else:
        # Positive exit codes indicate errors
        return f'error_code_{exit_code}'
```

**Concurrent Read Pattern:**

```python
def background_reader(pipe, result_container):
    """Runs in separate thread to prevent blocking"""
    try:
        while pipe.poll(timeout=0.1):
            chunk = pipe.recv()
            result_container.append(chunk)
    except EOFError:
        # Pipe closed, child process exited
        pass

# Start reader thread
results = []
reader = threading.Thread(target=background_reader, args=(pipe, results))
reader.start()

# Wait for process with timeout
status = wait_with_timeout(process, timeout=60)

# Collect results
reader.join(timeout=1.0)
return results, status
```

**Error Handling Matrix:**

| Exit Condition | Exit Code | Interpretation | Result |
|----------------|-----------|----------------|---------|
| Normal exit | 0 | Success | Return results |
| Python exception | 1 | Code error | Return exception info |
| SIGTERM | -15 | Timeout (graceful) | Return timeout error |
| SIGKILL | -9 | Timeout (forced) | Return timeout error |
| SIGSEGV | -11 | Crash | Return crash error |
| SIGABRT | -6 | Assertion failure | Return crash error |

**Resource Cleanup:**

```python
try:
    result = execute_in_subprocess(code, items)
finally:
    # Ensure subprocess is terminated
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()

    # Close pipes
    parent_conn.close()
    child_conn.close()
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_execute_process]]

=== Related Principles ===
* [[Principle:n8n-io_n8n_Subprocess_Isolation]] - Creates processes that produce results
* [[Principle:n8n-io_n8n_Code_Execution]] - Generates results within subprocess
* [[Principle:n8n-io_n8n_Task_Completion]] - Uses collected results for completion message
