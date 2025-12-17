{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Task_Management]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for receiving and processing task settings in the n8n Python task runner, initiating task execution after validation.

=== Description ===

The `_handle_task_settings` method is an asynchronous message handler within the `TaskRunner` class that processes incoming `BrokerTaskSettings` messages. It validates task existence, updates task state with workflow metadata, and spawns the task execution process.

=== Usage ===

This implementation is invoked automatically when the task runner receives settings from the n8n broker after initial task allocation. It serves as the transition point from task preparation to actual execution, ensuring the task exists and is properly configured before execution begins.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L282-300

=== Signature ===
<syntaxhighlight lang="python">
async def _handle_task_settings(self, message: BrokerTaskSettings) -> None:
    """Handle task settings received from broker."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_runner import TaskRunner
from src.broker_message import BrokerTaskSettings
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| message || BrokerTaskSettings || Yes || Message containing task ID and execution settings including workflow name and security configuration
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (none) || None || Method returns None but updates task state and spawns execution task
|}

=== Side Effects ===
* Updates `task_state.workflow_name` with workflow identifier
* Sets `task_state.status` to `TaskStatus.RUNNING`
* Creates asynchronous task for `_execute_task()`

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| TaskMissingError || Raised when `message.task_id` is not found in `running_tasks` dictionary
|}

== Implementation Details ==

=== Task State Validation ===
The method first retrieves the task state from the `running_tasks` dictionary using the task ID from the incoming message. If the task does not exist, it raises a `TaskMissingError`, preventing execution of unallocated tasks.

=== State Updates ===
Once validated, the method updates two critical pieces of task state:
* '''workflow_name''': Stored for logging and debugging purposes
* '''status''': Transitioned to `RUNNING` to indicate active execution

=== Asynchronous Execution ===
The method uses `asyncio.create_task()` to spawn task execution in the background, allowing the message handler to return immediately and continue processing other messages without blocking.

== Usage Examples ==

=== Internal Message Handling ===
<syntaxhighlight lang="python">
# This method is called internally by the TaskRunner's message dispatcher
# when a BrokerTaskSettings message is received

task_runner = TaskRunner()

# Message received from broker
settings_message = BrokerTaskSettings(
    task_id="task-123",
    settings=TaskSettings(
        workflow_name="data_validation_workflow",
        security_config=SecurityConfig(...)
    )
)

# Handler invoked by message router
await task_runner._handle_task_settings(settings_message)
# Task execution now running in background
</syntaxhighlight>

=== Error Handling Pattern ===
<syntaxhighlight lang="python">
try:
    await task_runner._handle_task_settings(settings_message)
except TaskMissingError as e:
    # Task was not properly allocated before settings received
    logger.error(f"Received settings for unknown task: {e.task_id}")
    # Send error response to broker
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Task_Settings_Reception]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[related::Implementation:n8n-io_n8n_TaskRunner_execute_task]]

=== Used By Workflow ===
* [[used_by::Workflow:n8n-io_n8n_Security_Validation]]
