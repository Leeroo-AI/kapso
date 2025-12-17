{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Automation]], [[domain::Task_Execution]], [[domain::Workflow_Orchestration]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for orchestrating the complete task execution lifecycle from validation through result reporting, provided by the n8n Python task runner.

=== Description ===

The `_execute_task()` async method coordinates the entire task execution workflow. It validates code security, creates and executes the subprocess, collects results and print statements, sends RPC messages for console logging, reports success or failure back to the broker, and handles cleanup. This method represents the highest-level task execution orchestration.

=== Usage ===

This implementation is invoked asynchronously after the runner accepts a task and receives its settings from the broker. It serves as the main coordinator that ties together validation, execution, result collection, and broker communication for a single task.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L302-371

=== Signature ===
<syntaxhighlight lang="python">
async def _execute_task(self, task_id: str, task_settings: TaskSettings) -> None:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_runner import TaskRunner
from src.message_types.broker import TaskSettings
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| task_id || str || Yes || Unique identifier for the task
|-
| task_settings.code || str || Yes || Python code to execute
|-
| task_settings.node_mode || NodeMode || Yes || "all_items" or "per_item"
|-
| task_settings.items || Items || Yes || Input data items
|-
| task_settings.query || Query || No || Query parameters (all_items mode)
|-
| task_settings.continue_on_fail || bool || Yes || Error handling strategy
|-
| task_settings.workflow_name || str || Yes || Workflow name for logging
|-
| task_settings.workflow_id || str || Yes || Workflow ID for logging
|-
| task_settings.node_name || str || Yes || Node name for logging
|-
| task_settings.node_id || str || Yes || Node ID for logging
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| RunnerTaskDone || Message || Sent on successful execution
|-
| RunnerTaskError || Message || Sent on task failure
|-
| RunnerRpcCall || Message || Sent for each print statement (console.log)
|}

== Usage Examples ==

=== Complete Task Execution ===
<syntaxhighlight lang="python">
import asyncio
from src.task_runner import TaskRunner
from src.message_types.broker import TaskSettings

async def main():
    runner = TaskRunner(config)

    # Task settings received from broker
    task_settings = TaskSettings(
        code="""
result = []
for item in _items:
    value = item["json"]["value"]
    print(f"Processing: {value}")
    result.append({"json": {"doubled": value * 2}})
return result
        """,
        node_mode="all_items",
        items=[
            {"json": {"value": 10}},
            {"json": {"value": 20}}
        ],
        continue_on_fail=False,
        workflow_name="My Workflow",
        workflow_id="wf_123",
        node_name="Python Node",
        node_id="node_456"
    )

    # Execute task (this is called internally by message handler)
    await runner._execute_task("task_789", task_settings)

    # Runner sends:
    # 1. RunnerRpcCall for "Processing: 10"
    # 2. RunnerRpcCall for "Processing: 20"
    # 3. RunnerTaskDone with result data

asyncio.run(main())
</syntaxhighlight>

=== Error Handling Example ===
<syntaxhighlight lang="python">
# Task with invalid code
task_settings = TaskSettings(
    code="import os; os.system('rm -rf /')",  # Security violation
    node_mode="all_items",
    items=[{"json": {"value": 1}}],
    continue_on_fail=False,
    workflow_name="Test",
    workflow_id="wf_test",
    node_name="Bad Code",
    node_id="node_test"
)

# Execution fails during validation
# Runner sends: RunnerTaskError with security violation details
await runner._execute_task("task_bad", task_settings)
</syntaxhighlight>

== Implementation Details ==

=== Execution Timing ===
<syntaxhighlight lang="python">
start_time = time.time()

# ... execution ...

LOG_TASK_COMPLETE.format(
    task_id=task_id,
    duration=self._get_duration(start_time),
    result_size=self._get_result_size(result_size_bytes),
    **task_state.context(),
)
</syntaxhighlight>

Duration tracking for:
* Performance monitoring
* Timeout analysis
* Workflow optimization insights

=== Task State Validation ===
<syntaxhighlight lang="python">
task_state = self.running_tasks.get(task_id)

if task_state is None:
    raise TaskMissingError(task_id)
</syntaxhighlight>

Ensures task was properly accepted before execution begins.

=== Security Validation ===
<syntaxhighlight lang="python">
self.analyzer.validate(task_settings.code)
</syntaxhighlight>

AST-based validation before execution:
* Checks import allowlists
* Blocks dangerous attribute access
* Prevents relative imports
* Validates dynamic imports

Raises `SecurityViolationError` if code is unsafe.

=== Process Creation and Execution ===
<syntaxhighlight lang="python">
process, read_conn, write_conn = self.executor.create_process(
    code=task_settings.code,
    node_mode=task_settings.node_mode,
    items=task_settings.items,
    security_config=self.security_config,
    query=task_settings.query,
)

task_state.process = process  # Store for cancellation

result, print_args, result_size_bytes = await asyncio.to_thread(
    self.executor.execute_process,
    process=process,
    read_conn=read_conn,
    write_conn=write_conn,
    task_timeout=self.config.task_timeout,
    pipe_reader_timeout=self.config.pipe_reader_timeout,
    continue_on_fail=task_settings.continue_on_fail,
)
</syntaxhighlight>

Key points:
* Subprocess creation happens in main thread
* Execution delegated to thread pool (blocking operations)
* Process handle stored for cancellation support
* All subprocess I/O handled in worker thread

=== Print Statement Processing ===
<syntaxhighlight lang="python">
for print_args_per_call in print_args:
    await self._send_rpc_message(
        task_id, RPC_BROWSER_CONSOLE_LOG_METHOD, print_args_per_call
    )
</syntaxhighlight>

Print statements are sent as RPC calls:
* Method: "console.log" (browser console API)
* Each print() call becomes one RPC message
* Formatted for JavaScript console display
* Sent before result message

RPC message structure:
<syntaxhighlight lang="python">
RunnerRpcCall(
    call_id=nanoid(),
    task_id=task_id,
    name="console.log",
    params=["'Processing: 10'"]  # Formatted args
)
</syntaxhighlight>

=== Success Response ===
<syntaxhighlight lang="python">
response = RunnerTaskDone(task_id=task_id, data={"result": result})
await self._send_message(response)

self.logger.info(
    LOG_TASK_COMPLETE.format(
        task_id=task_id,
        duration=self._get_duration(start_time),
        result_size=self._get_result_size(result_size_bytes),
        **task_state.context(),
    )
)
</syntaxhighlight>

Success logging includes:
* Task ID
* Execution duration (ms, s, or m)
* Result size (bytes, KB, or MB)
* Workflow and node context

=== Cancellation Handling ===
<syntaxhighlight lang="python">
except TaskCancelledError as e:
    response = RunnerTaskError(task_id=task_id, error={"message": str(e)})
    await self._send_message(response)
</syntaxhighlight>

Cancellation occurs when:
* Broker sends BrokerTaskCancel message
* Task timeout exceeded (SIGTERM → SIGKILL)
* Runner shutdown requested

=== Syntax Error Handling ===
<syntaxhighlight lang="python">
except SyntaxError as e:
    self.logger.warning(f"Task {task_id} failed syntax validation")
    error = {"message": str(e)}
    response = RunnerTaskError(task_id=task_id, error=error)
    await self._send_message(response)
</syntaxhighlight>

Syntax errors caught early:
* Code compilation failures
* Python syntax violations
* Invalid code structure

=== General Error Handling ===
<syntaxhighlight lang="python">
except Exception as e:
    self.logger.error(f"Task {task_id} failed", exc_info=True)
    error = {
        "message": getattr(e, "message", str(e)),
        "description": getattr(e, "description", ""),
    }
    response = RunnerTaskError(task_id=task_id, error=error)
    await self._send_message(response)
</syntaxhighlight>

Comprehensive error details:
* Exception message (primary error)
* Description (additional context if available)
* Full stack trace logged locally

=== Cleanup and Idle Timer ===
<syntaxhighlight lang="python">
finally:
    self.running_tasks.pop(task_id, None)
    self._reset_idle_timer()
</syntaxhighlight>

Cleanup always occurs:
* Remove task from running_tasks dict
* Reset idle timer (for auto-shutdown)
* Free capacity for new offers

Idle timer reset is critical:
* Marks completion as key activity event
* Prevents premature shutdown
* Allows runner to accept new tasks

=== Context Logging ===
<syntaxhighlight lang="python">
task_state.context()
# Returns:
{
    "workflow_name": "My Workflow",
    "workflow_id": "wf_123",
    "node_name": "Python Node",
    "node_id": "node_456"
}
</syntaxhighlight>

Context used for structured logging:
* Workflow identification
* Node identification
* Debugging and tracing
* Performance analysis

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Task_Completion]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[Implementation:n8n-io_n8n_TaskAnalyzer_validate]]
* [[Implementation:n8n-io_n8n_TaskExecutor_create_process]]
* [[Implementation:n8n-io_n8n_TaskExecutor_execute_process]]
* [[Implementation:n8n-io_n8n_TaskRunner_handle_task_offer_accept]]

=== Execution Flow ===
# Validate task state exists
# Validate code security (AST analysis)
# Create subprocess with settings
# Execute in thread pool (blocking operations)
# Collect results and print statements
# Send print statements via RPC
# Send success/error response
# Log completion with metrics
# Clean up task state
# Reset idle timer

=== Error Categories ===
{| class="wikitable"
|-
! Error Type !! Handling !! Logged As
|-
| TaskCancelledError || Send error response || Info
|-
| SyntaxError || Send error response || Warning
|-
| SecurityViolationError || Send error response || Warning
|-
| Other exceptions || Send error response || Error (with stack)
|}

=== Performance Metrics ===
* '''Duration:''' Formatted as ms, s, or m
* '''Result size:''' Formatted as bytes, KB, or MB
* '''Start time:''' Captured before execution
* '''End time:''' Implicit in logging

=== Message Sequence ===
# (Input) BrokerTaskSettings → Runner
# (Output) RunnerRpcCall (× N print statements) → Broker
# (Output) RunnerTaskDone or RunnerTaskError → Broker
