{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Automation]], [[domain::Task_Execution]], [[domain::WebSocket]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for establishing and maintaining a WebSocket connection to the n8n broker, provided by the n8n Python task runner.

=== Description ===

The `start()` method is the main entry point of the `TaskRunner` class. It establishes a WebSocket connection to the task broker with bearer token authentication, handles connection failures with automatic retry logic, and listens for incoming messages. The method runs in a continuous loop until shutdown is requested.

=== Usage ===

This implementation is invoked when the Python task runner process starts. It manages the lifecycle of the WebSocket connection, including reconnection attempts after failures, and serves as the foundation for all broker-runner communication in the task execution workflow.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L115-146

=== Signature ===
<syntaxhighlight lang="python">
async def start(self) -> None:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_runner import TaskRunner
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| self.config.grant_token || str || Yes || Bearer token for WebSocket authentication
|-
| self.config.is_auto_shutdown_enabled || bool || Yes || Flag indicating if auto-shutdown is enabled
|-
| self.config.max_payload_size || int || Yes || Maximum WebSocket message size
|-
| self.on_idle_timeout || Callable || Conditional || Required if auto_shutdown is enabled
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Method runs until shutdown; communicates via WebSocket
|-
| Exception || NoIdleTimeoutHandlerError || Raised if auto-shutdown enabled without handler
|-
| Exception || InvalidStatus || Raised on authentication failure (status 403)
|}

== Usage Examples ==

=== Starting the Task Runner ===
<syntaxhighlight lang="python">
import asyncio
from src.task_runner import TaskRunner
from src.config.task_runner_config import TaskRunnerConfig

async def main():
    config = TaskRunnerConfig(
        grant_token="your_bearer_token",
        task_broker_uri="http://localhost:5679",
        max_concurrency=5,
        max_payload_size=1024 * 1024,  # 1MB
        is_auto_shutdown_enabled=False
    )

    runner = TaskRunner(config)

    # Start the runner - this blocks until shutdown
    await runner.start()

if __name__ == "__main__":
    asyncio.run(main())
</syntaxhighlight>

=== With Auto-Shutdown Handler ===
<syntaxhighlight lang="python">
import asyncio
from src.task_runner import TaskRunner
from src.config.task_runner_config import TaskRunnerConfig

async def handle_idle_timeout():
    print("Runner has been idle, shutting down...")
    # Cleanup logic here

async def main():
    config = TaskRunnerConfig(
        grant_token="your_token",
        task_broker_uri="http://localhost:5679",
        max_concurrency=5,
        is_auto_shutdown_enabled=True,
        auto_shutdown_timeout=300  # 5 minutes
    )

    runner = TaskRunner(config)
    runner.on_idle_timeout = handle_idle_timeout

    await runner.start()

asyncio.run(main())
</syntaxhighlight>

== Implementation Details ==

=== Connection Establishment ===
The method constructs WebSocket connection headers with bearer token authentication:
<syntaxhighlight lang="python">
headers = {"Authorization": f"Bearer {self.config.grant_token}"}
</syntaxhighlight>

=== Retry Logic ===
Connection failures trigger automatic retry with 5-second delay:
* Authentication failures (status 403) are logged as errors and re-raised
* Other connection failures are logged as warnings and retried indefinitely
* On reconnection, offer sending is disabled and coroutines are cancelled

=== Connection Loop ===
The main loop continues until `is_shutting_down` flag is set:
<syntaxhighlight lang="python">
while not self.is_shutting_down:
    try:
        self.websocket_connection = await websockets.connect(...)
        await self._listen_for_messages()
    except Exception:
        await asyncio.sleep(5)  # Retry delay
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_WebSocket_Connection]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[Implementation:n8n-io_n8n_TaskRunner_send_offers]]
* [[Implementation:n8n-io_n8n_TaskRunner_handle_task_offer_accept]]

=== Dependencies ===
* websockets library for async WebSocket client
* TaskRunnerConfig for configuration management
* WebSocket URL constructed as: ws://{host}/task-broker/runner?id={runner_id}
