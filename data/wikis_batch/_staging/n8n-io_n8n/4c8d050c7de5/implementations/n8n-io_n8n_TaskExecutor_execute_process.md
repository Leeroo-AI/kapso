{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Automation]], [[domain::Task_Execution]], [[domain::Process_Management]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for managing subprocess execution lifecycle, timeouts, and result collection, provided by the n8n Python task runner.

=== Description ===

The `execute_process()` static method orchestrates the complete lifecycle of subprocess execution. It starts the process, manages timeout constraints, reads results from the pipe using a dedicated reader thread, handles various failure modes (timeout, termination, kill signals), and returns the execution results with metadata.

=== Usage ===

This implementation is invoked by the task runner after creating a subprocess with `create_process()`. It handles all aspects of process supervision, including timeout enforcement, result collection, error handling, and proper cleanup of resources.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_executor.py
* '''Lines:''' L88-165

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def execute_process(
    process: ForkServerProcess,
    read_conn: PipeConnection,
    write_conn: PipeConnection,
    task_timeout: int,
    pipe_reader_timeout: float,
    continue_on_fail: bool,
) -> tuple[Items, PrintArgs, int]:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from multiprocessing.context import ForkServerProcess
from multiprocessing.connection import Connection
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| process || ForkServerProcess || Yes || Subprocess to execute
|-
| read_conn || PipeConnection || Yes || Read end of communication pipe
|-
| write_conn || PipeConnection || Yes || Write end of communication pipe
|-
| task_timeout || int || Yes || Maximum execution time in seconds
|-
| pipe_reader_timeout || float || Yes || Maximum time to wait for pipe data
|-
| continue_on_fail || bool || Yes || Whether to return error items on failure
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || Items || List of result items from execution
|-
| print_args || PrintArgs || Captured print statements
|-
| result_size_bytes || int || Size of result message in bytes
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Raised When
|-
| TaskSubprocessFailedError || Process failed to start or exited with non-zero code
|-
| TaskTimeoutError || Execution exceeded task_timeout
|-
| TaskCancelledError || Process terminated with SIGTERM
|-
| TaskKilledError || Process terminated with SIGKILL
|-
| TaskResultReadError || Error reading from pipe
|-
| TaskResultMissingError || No result data received
|-
| TaskRuntimeError || User code raised exception
|}

== Usage Examples ==

=== Complete Task Execution Flow ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.config.security_config import SecurityConfig

code = """
result = []
for item in _items:
    result.append({"json": {"value": item["json"]["value"] * 2}})
return result
"""

items = [{"json": {"value": 10}}]
security_config = SecurityConfig(
    stdlib_allow=["json"],
    external_allow=[],
    builtins_deny=["eval", "exec"],
    runner_env_deny=True
)

# Step 1: Create process
process, read_conn, write_conn = TaskExecutor.create_process(
    code=code,
    node_mode="all_items",
    items=items,
    security_config=security_config
)

# Step 2: Execute process
try:
    result, print_args, result_size = TaskExecutor.execute_process(
        process=process,
        read_conn=read_conn,
        write_conn=write_conn,
        task_timeout=30,
        pipe_reader_timeout=2.0,
        continue_on_fail=False
    )

    print(f"Result: {result}")
    print(f"Print statements: {print_args}")
    print(f"Result size: {result_size} bytes")

except TaskTimeoutError as e:
    print(f"Task timed out after {e.timeout} seconds")
except TaskRuntimeError as e:
    print(f"User code error: {e.message}")
</syntaxhighlight>

=== Handling Continue-on-Fail ===
<syntaxhighlight lang="python">
# With continue_on_fail=True, errors become data
result, print_args, result_size = TaskExecutor.execute_process(
    process=process,
    read_conn=read_conn,
    write_conn=write_conn,
    task_timeout=30,
    pipe_reader_timeout=2.0,
    continue_on_fail=True  # Errors returned as items
)

# On error, returns:
# [{"json": {"error": "Error message"}}]
# Instead of raising exception
</syntaxhighlight>

=== Manual Process Stopping ===
<syntaxhighlight lang="python">
# Stop a running process gracefully
process, read_conn, write_conn = TaskExecutor.create_process(...)
process.start()

# Later, stop the process
TaskExecutor.stop_process(process)
# Sends SIGTERM, waits 1s, then SIGKILL if needed
</syntaxhighlight>

== Implementation Details ==

=== Pipe Reader Thread ===
<syntaxhighlight lang="python">
pipe_reader = PipeReader(read_conn.fileno(), read_conn)
pipe_reader.start()
</syntaxhighlight>

The PipeReader runs in a separate thread to:
* Read length-prefixed messages asynchronously
* Avoid blocking the main execution thread
* Handle large result payloads efficiently
* Capture errors during read operations

=== Process Startup ===
<syntaxhighlight lang="python">
try:
    try:
        process.start()
    except Exception as e:
        raise TaskSubprocessFailedError(-1, e)
    finally:
        write_conn.close()  # Subprocess has its own copy
</syntaxhighlight>

Key points:
* Write connection closed in parent process immediately
* Subprocess retains its copy for writing results
* Startup errors wrapped in TaskSubprocessFailedError

=== Timeout Enforcement ===
<syntaxhighlight lang="python">
process.join(timeout=task_timeout)

if process.is_alive():
    TaskExecutor.stop_process(process)
    raise TaskTimeoutError(task_timeout)
</syntaxhighlight>

Timeout handling:
* `join()` waits up to task_timeout seconds
* If still alive after timeout, process is stopped
* stop_process() sends SIGTERM, then SIGKILL after 1s grace period

=== Exit Code Interpretation ===
<syntaxhighlight lang="python">
if process.exitcode == SIGTERM_EXIT_CODE:  # -15
    raise TaskCancelledError()

if process.exitcode == SIGKILL_EXIT_CODE:  # -9
    raise TaskKilledError()

if process.exitcode != 0:
    assert process.exitcode is not None
    raise TaskSubprocessFailedError(process.exitcode)
</syntaxhighlight>

Exit code meanings:
* '''0:''' Success
* '''-15 (SIGTERM):''' Graceful cancellation requested
* '''-9 (SIGKILL):''' Forceful termination
* '''Other non-zero:''' Subprocess failure

=== Pipe Reader Timeout ===
<syntaxhighlight lang="python">
pipe_reader.join(timeout=pipe_reader_timeout)

if pipe_reader.is_alive():
    logger.warning(
        LOG_PIPE_READER_TIMEOUT_TRIGGERED.format(
            timeout=pipe_reader_timeout
        )
    )
    try:
        read_conn.close()
    except Exception:
        pass
</syntaxhighlight>

Pipe reader timeout scenarios:
* Subprocess finished but data still being read
* Large result payload taking time to transfer
* Pipe buffer filling slowly
* If timeout occurs, connection is forcefully closed

=== Result Validation ===
<syntaxhighlight lang="python">
if pipe_reader.error:
    raise TaskResultReadError(pipe_reader.error)

if pipe_reader.pipe_message is None:
    raise TaskResultMissingError()

returned = pipe_reader.pipe_message

if "error" in returned:
    raise TaskRuntimeError(returned["error"])

if "result" not in returned:
    raise TaskResultMissingError()
</syntaxhighlight>

Validation checks:
1. Pipe reader encountered no I/O errors
2. Message was successfully received
3. Message contains either "result" or "error"
4. If error present, raise TaskRuntimeError
5. If no result field, data is malformed

=== Result Extraction ===
<syntaxhighlight lang="python">
result = returned["result"]
print_args = returned.get("print_args", [])
assert pipe_reader.message_size is not None
result_size_bytes = pipe_reader.message_size

return result, print_args, result_size_bytes
</syntaxhighlight>

Return value includes:
* '''result:''' List of output items
* '''print_args:''' Captured print statements
* '''result_size_bytes:''' Total message size for logging

=== Continue-on-Fail Handling ===
<syntaxhighlight lang="python">
except Exception as e:
    if continue_on_fail:
        return [{"json": {"error": str(e)}}], print_args, 0
    raise
</syntaxhighlight>

When continue_on_fail is True:
* Exceptions converted to error items
* Workflow continues despite task failure
* Print args still returned for debugging
* Result size reported as 0

=== Process Stop Implementation ===
<syntaxhighlight lang="python">
@staticmethod
def stop_process(process: ForkServerProcess | None):
    if process is None or not process.is_alive():
        return

    try:
        process.terminate()  # Send SIGTERM
        process.join(timeout=1)  # 1s grace period

        if process.is_alive():
            process.kill()  # Send SIGKILL
            process.join()
    except (ProcessLookupError, ConnectionError, BrokenPipeError):
        pass  # Process already dead
</syntaxhighlight>

Two-phase shutdown:
1. '''Graceful (SIGTERM):''' Allow cleanup, 1 second grace period
2. '''Forceful (SIGKILL):''' Immediate termination if still alive

Exceptions ignored:
* ProcessLookupError: Process already exited
* ConnectionError: Pipe already closed
* BrokenPipeError: Communication channel broken

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Result_Collection]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[Implementation:n8n-io_n8n_TaskExecutor_create_process]]
* [[Implementation:n8n-io_n8n_TaskExecutor_all_items]]
* [[Implementation:n8n-io_n8n_TaskRunner_execute_task]]

=== Timeout Management ===
{| class="wikitable"
|-
! Timeout !! Purpose !! Typical Value
|-
| task_timeout || Maximum code execution time || 30-300 seconds
|-
| pipe_reader_timeout || Maximum time to read results || 2.0 seconds
|-
| stop_process grace period || Time before SIGKILL || 1.0 second
|}

=== Exit Codes ===
{| class="wikitable"
|-
! Code !! Signal !! Meaning !! Exception
|-
| 0 || None || Success || None
|-
| -15 || SIGTERM || Graceful cancellation || TaskCancelledError
|-
| -9 || SIGKILL || Forceful termination || TaskKilledError
|-
| Other || Various || Subprocess failure || TaskSubprocessFailedError
|}

=== Error Recovery ===
* Process startup errors: Wrapped and reported
* Timeout errors: Process stopped, error raised
* Pipe read errors: Reported with context
* Runtime errors: Stack trace included
* Continue-on-fail: Errors converted to data items
