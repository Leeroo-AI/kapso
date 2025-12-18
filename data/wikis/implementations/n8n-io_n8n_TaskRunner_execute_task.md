# Implementation: TaskRunner._execute_task

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Error_Handling]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete async method for orchestrating complete task execution lifecycle from validation through result delivery.

=== Description ===

`TaskRunner._execute_task()` is the central orchestration point for task execution:

1. **Task State Lookup**: Retrieves task state from `running_tasks`
2. **Security Validation**: Calls `analyzer.validate()` on task code
3. **Process Creation**: Creates subprocess via `executor.create_process()`
4. **Async Execution**: Runs `execute_process` in thread pool for non-blocking wait
5. **Print Forwarding**: Sends captured print() calls via RPC to browser console
6. **Success Response**: Sends `RunnerTaskDone` with result data
7. **Error Handling**: Catches and reports various error types
8. **Cleanup**: Always removes task and resets idle timer in `finally` block

=== Usage ===

This method is called internally when `BrokerTaskSettings` message is received. It runs as an asyncio task, allowing the runner to handle multiple tasks concurrently.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L302-371

=== Signature ===
<syntaxhighlight lang="python">
async def _execute_task(self, task_id: str, task_settings: TaskSettings) -> None:
    """
    Execute a task and send result to broker.

    Args:
        task_id: Unique identifier for the task.
        task_settings: Task configuration including code, items, mode, etc.

    Sends:
        RunnerTaskDone: On successful execution with result data.
        RunnerTaskError: On any error with message and description.

    Side Effects:
        - Removes task from self.running_tasks
        - Resets idle timer
        - Forwards print statements via RPC
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal method - accessed via TaskRunner instance
from src.task_runner import TaskRunner
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| task_id || str || Yes || Unique task identifier
|-
| task_settings || TaskSettings || Yes || Contains code, items, node_mode, continue_on_fail, query, workflow metadata
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (WebSocket) || RunnerTaskDone || Sent on success with {"result": items}
|-
| (WebSocket) || RunnerTaskError || Sent on failure with {"message": ..., "description": ...}
|-
| (WebSocket) || RunnerRpcCall || Print statements forwarded via "console.log" RPC
|}

=== TaskSettings Structure ===
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| code || str || Python source code to execute
|-
| items || Items || Input data items
|-
| node_mode || NodeMode || "all_items" or "per_item"
|-
| continue_on_fail || bool || Return error in result instead of failing
|-
| query || Query || Optional query data
|-
| workflow_name || str || Workflow name for logging
|-
| workflow_id || str || Workflow ID for logging
|-
| node_name || str || Node name for logging
|-
| node_id || str || Node ID for logging
|}

== Usage Examples ==

=== Execution Flow ===
<syntaxhighlight lang="python">
# Internal execution flow (simplified)

async def _execute_task(self, task_id, task_settings):
    start_time = time.time()

    try:
        # 1. Validate security
        self.analyzer.validate(task_settings.code)

        # 2. Create process
        process, read_conn, write_conn = self.executor.create_process(
            code=task_settings.code,
            node_mode=task_settings.node_mode,
            items=task_settings.items,
            security_config=self.security_config,
            query=task_settings.query,
        )

        task_state.process = process

        # 3. Execute in thread (non-blocking)
        result, print_args, size = await asyncio.to_thread(
            self.executor.execute_process,
            process=process,
            read_conn=read_conn,
            write_conn=write_conn,
            task_timeout=self.config.task_timeout,
            pipe_reader_timeout=self.config.pipe_reader_timeout,
            continue_on_fail=task_settings.continue_on_fail,
        )

        # 4. Forward prints to browser
        for args in print_args:
            await self._send_rpc_message(task_id, "console.log", args)

        # 5. Send success
        await self._send_message(RunnerTaskDone(task_id, {"result": result}))

    except TaskCancelledError as e:
        await self._send_message(RunnerTaskError(task_id, {"message": str(e)}))

    except SecurityViolationError as e:
        await self._send_message(RunnerTaskError(task_id, {
            "message": e.message,
            "description": e.description,
        }))

    finally:
        self.running_tasks.pop(task_id, None)
        self._reset_idle_timer()
</syntaxhighlight>

=== Logging Output ===
<syntaxhighlight lang="python">
# Success log format
LOG_TASK_COMPLETE = (
    "Task {task_id} completed in {duration} "
    "(result: {result_size}, workflow: {workflow_name}, node: {node_name})"
)

# Example:
# Task abc123 completed in 250ms (result: 1.2 KB, workflow: My Workflow, node: Python)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Result_Delivery]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
