# Implementation: TaskRunner.start

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Network_Communication]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete async method for establishing and maintaining a WebSocket connection to the task broker with automatic reconnection.

=== Description ===

`TaskRunner.start()` is the main entry point for running the task runner service. It:

1. Validates auto-shutdown configuration
2. Establishes a WebSocket connection with bearer token authentication
3. Configures maximum payload size for large task data
4. Starts the message listener coroutine
5. Handles connection failures with automatic retry (5 second delay)
6. Gracefully handles authentication failures (403 responses)

The method runs in a loop until `is_shutting_down` is set, automatically reconnecting on transient failures.

=== Usage ===

Call this method after creating a TaskRunner instance to start processing tasks. Typically used with `asyncio.run()` or as part of an async main function.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L115-146

=== Signature ===
<syntaxhighlight lang="python">
async def start(self) -> None:
    """
    Start the task runner, connecting to the broker.

    Raises:
        NoIdleTimeoutHandlerError: If auto-shutdown is enabled but no handler is set.
        InvalidStatus: If authentication fails with 403 status.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_runner import TaskRunner
import asyncio
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (uses self) || - || - || Uses config.grant_token, config.max_payload_size, websocket_url from __init__
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| self.websocket_connection || ClientConnection || Active WebSocket connection to broker
|-
| self.offers_coroutine || asyncio.Task || Background task sending periodic offers
|-
| (logs) || - || Connection status logged via self.logger
|}

=== Side Effects ===
* Establishes network connection to broker
* Starts background coroutine for message listening
* Sets `self.can_send_offers = True` upon successful registration

== Usage Examples ==

=== Basic Startup ===
<syntaxhighlight lang="python">
import asyncio
from src.task_runner import TaskRunner
from src.config.task_runner_config import TaskRunnerConfig

async def main():
    config = TaskRunnerConfig.from_env()
    runner = TaskRunner(config)

    try:
        await runner.start()
    except KeyboardInterrupt:
        await runner.stop()

asyncio.run(main())
</syntaxhighlight>

=== With Auto-Shutdown Handler ===
<syntaxhighlight lang="python">
import asyncio
from src.task_runner import TaskRunner
from src.config.task_runner_config import TaskRunnerConfig

async def main():
    config = TaskRunnerConfig.from_env()
    runner = TaskRunner(config)

    # Set idle timeout handler for auto-shutdown
    async def handle_idle_timeout():
        print("Runner idle - shutting down")
        await runner.stop()

    runner.on_idle_timeout = handle_idle_timeout

    await runner.start()

asyncio.run(main())
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Broker_Connection]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
