{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Automation]], [[domain::Task_Execution]], [[domain::Load_Balancing]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for generating and sending capacity-based task offers to the broker, provided by the n8n Python task runner.

=== Description ===

The `_send_offers_loop()` and `_send_offers()` methods work together to continuously advertise the task runner's availability to the broker. The loop runs at regular intervals, calculating available capacity and sending offers with validity windows and jitter to prevent thundering herd problems.

=== Usage ===

These implementations are automatically invoked after the runner successfully registers with the broker. They maintain a pool of open offers that the broker can accept, enabling dynamic task distribution based on runner capacity.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L431-473

=== Signature ===
<syntaxhighlight lang="python">
async def _send_offers_loop(self) -> None:

async def _send_offers(self) -> None:
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
| self.can_send_offers || bool || Yes || Flag enabling offer generation
|-
| self.config.max_concurrency || int || Yes || Maximum number of concurrent tasks
|-
| self.open_offers || dict[str, TaskOffer] || Yes || Dictionary of pending offers
|-
| self.running_tasks || dict[str, TaskState] || Yes || Dictionary of active tasks
|-
| OFFER_INTERVAL || float || Yes || Seconds between offer batches (constant)
|-
| OFFER_VALIDITY || int || Yes || Base validity duration in milliseconds
|-
| OFFER_VALIDITY_MAX_JITTER || int || Yes || Maximum random jitter in milliseconds
|-
| OFFER_VALIDITY_LATENCY_BUFFER || float || Yes || Buffer time for network latency
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| RunnerTaskOffer || Message || WebSocket message sent to broker with offer details
|-
| self.open_offers || dict || Updated with newly created offers
|}

== Usage Examples ==

=== Automatic Offer Loop ===
<syntaxhighlight lang="python">
import asyncio
from src.task_runner import TaskRunner
from src.config.task_runner_config import TaskRunnerConfig

async def main():
    config = TaskRunnerConfig(
        grant_token="token",
        task_broker_uri="http://localhost:5679",
        max_concurrency=10  # Will send up to 10 offers
    )

    runner = TaskRunner(config)

    # Offer loop starts automatically after registration
    await runner.start()

asyncio.run(main())
</syntaxhighlight>

=== Manual Offer Sending ===
<syntaxhighlight lang="python">
# Internal usage when task is cancelled while waiting for settings
async def _handle_task_cancel(self, message):
    if task_state.status == TaskStatus.WAITING_FOR_SETTINGS:
        self.running_tasks.pop(task_id, None)
        # Immediately send new offer to replace cancelled task
        await self._send_offers()
</syntaxhighlight>

== Implementation Details ==

=== Capacity Calculation ===
The number of offers to send is calculated dynamically:
<syntaxhighlight lang="python">
offers_to_send = self.config.max_concurrency - (
    len(self.open_offers) + self.running_tasks_count
)
</syntaxhighlight>

This ensures total capacity (offers + running tasks) never exceeds max_concurrency.

=== Offer Validity with Jitter ===
Each offer includes a validity window with randomized jitter:
<syntaxhighlight lang="python">
valid_for_ms = OFFER_VALIDITY + random.randint(0, OFFER_VALIDITY_MAX_JITTER)

valid_until = (
    time.time() + (valid_for_ms / 1000) + OFFER_VALIDITY_LATENCY_BUFFER
)
</syntaxhighlight>

* '''Base validity:''' Consistent minimum lifetime
* '''Jitter:''' Prevents synchronized expiration across runners
* '''Latency buffer:''' Accounts for network delays

=== Expired Offer Cleanup ===
Before sending new offers, expired ones are removed:
<syntaxhighlight lang="python">
expired_offer_ids = [
    offer_id
    for offer_id, offer in self.open_offers.items()
    if offer.has_expired
]

for offer_id in expired_offer_ids:
    self.open_offers.pop(offer_id, None)
</syntaxhighlight>

=== Loop Error Handling ===
The loop continues running despite errors:
<syntaxhighlight lang="python">
while self.can_send_offers:
    try:
        await self._send_offers()
        await asyncio.sleep(OFFER_INTERVAL)
    except asyncio.CancelledError:
        break  # Clean shutdown
    except Exception as e:
        self.logger.error(f"Error sending offers: {e}")
        # Loop continues
</syntaxhighlight>

=== Offer Message Structure ===
<syntaxhighlight lang="python">
message = RunnerTaskOffer(
    offer_id=nanoid(),          # Unique offer identifier
    task_type=TASK_TYPE_PYTHON, # "python" type constant
    valid_for=valid_for_ms      # Milliseconds until expiration
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Offer_Based_Distribution]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[Implementation:n8n-io_n8n_TaskRunner_start]]
* [[Implementation:n8n-io_n8n_TaskRunner_handle_task_offer_accept]]

=== Constants Used ===
* OFFER_INTERVAL: Time between offer batches
* OFFER_VALIDITY: Base validity duration
* OFFER_VALIDITY_MAX_JITTER: Maximum random variance
* OFFER_VALIDITY_LATENCY_BUFFER: Network latency compensation
* TASK_TYPE_PYTHON: Task type identifier ("python")
