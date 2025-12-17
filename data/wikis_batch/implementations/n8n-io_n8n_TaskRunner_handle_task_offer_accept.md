{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Automation]], [[domain::Task_Execution]], [[domain::State_Management]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for validating broker acceptance of task offers and initializing task state, provided by the n8n Python task runner.

=== Description ===

The `_handle_task_offer_accept()` method processes broker acceptance messages for previously sent offers. It validates that the offer exists and hasn't expired, checks capacity constraints, creates task state tracking, and sends either acceptance or rejection responses back to the broker.

=== Usage ===

This implementation is invoked when the broker accepts one of the runner's task offers. It serves as the critical validation point before committing to execute a task, ensuring the runner can handle the workload and the offer is still valid.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L253-280

=== Signature ===
<syntaxhighlight lang="python">
async def _handle_task_offer_accept(self, message: BrokerTaskOfferAccept) -> None:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_runner import TaskRunner
from src.message_types import BrokerTaskOfferAccept
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| message.offer_id || str || Yes || ID of the offer being accepted
|-
| message.task_id || str || Yes || Unique task identifier assigned by broker
|-
| self.open_offers || dict[str, TaskOffer] || Yes || Dictionary of pending offers
|-
| self.running_tasks_count || int || Yes || Current number of running tasks
|-
| self.config.max_concurrency || int || Yes || Maximum concurrent task limit
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| RunnerTaskAccepted || Message || Sent when offer is valid and capacity available
|-
| RunnerTaskRejected || Message || Sent when offer expired or at capacity
|-
| self.running_tasks[task_id] || TaskState || New task state created on acceptance
|-
| self.open_offers || dict || Accepted offer removed from dictionary
|}

== Usage Examples ==

=== Normal Offer Acceptance Flow ===
<syntaxhighlight lang="python">
# This is automatically handled by the TaskRunner message loop
# Example of what happens internally:

# 1. Broker sends BrokerTaskOfferAccept message
broker_message = BrokerTaskOfferAccept(
    offer_id="offer_abc123",
    task_id="task_xyz789"
)

# 2. Runner processes acceptance
await runner._handle_task_offer_accept(broker_message)

# 3. If valid, creates task state
# runner.running_tasks["task_xyz789"] = TaskState("task_xyz789")

# 4. Sends acceptance confirmation
# RunnerTaskAccepted(task_id="task_xyz789")
</syntaxhighlight>

=== Checking Offer Validity ===
<syntaxhighlight lang="python">
from src.task_runner import TaskOffer
import time

# Offers have expiration timestamps
offer = TaskOffer(
    offer_id="offer_123",
    valid_until=time.time() + 30.0  # Valid for 30 seconds
)

# Check if expired
if offer.has_expired:
    # Reject with expiration reason
    response = RunnerTaskRejected(
        task_id=task_id,
        reason=TASK_REJECTED_REASON_OFFER_EXPIRED
    )
</syntaxhighlight>

== Implementation Details ==

=== Validation Sequence ===

'''Step 1: Offer Existence and Expiration Check'''
<syntaxhighlight lang="python">
offer = self.open_offers.get(message.offer_id)

if offer is None or offer.has_expired:
    response = RunnerTaskRejected(
        task_id=message.task_id,
        reason=TASK_REJECTED_REASON_OFFER_EXPIRED,
    )
    await self._send_message(response)
    return
</syntaxhighlight>

Rejection occurs if:
* Offer ID not found in open_offers dictionary
* Offer timestamp exceeds valid_until time

'''Step 2: Capacity Check'''
<syntaxhighlight lang="python">
if self.running_tasks_count >= self.config.max_concurrency:
    response = RunnerTaskRejected(
        task_id=message.task_id,
        reason=TASK_REJECTED_REASON_AT_CAPACITY,
    )
    await self._send_message(response)
    return
</syntaxhighlight>

This prevents race conditions where multiple offers are accepted simultaneously.

'''Step 3: Task State Initialization'''
<syntaxhighlight lang="python">
del self.open_offers[message.offer_id]

task_state = TaskState(message.task_id)
self.running_tasks[message.task_id] = task_state
</syntaxhighlight>

Task state is created with:
* Initial status: WAITING_FOR_SETTINGS
* Task ID from broker
* Empty workflow/node metadata (populated later)

'''Step 4: Acceptance Confirmation'''
<syntaxhighlight lang="python">
response = RunnerTaskAccepted(task_id=message.task_id)
await self._send_message(response)
self.logger.info(f"Accepted task {message.task_id}")
self._reset_idle_timer()
</syntaxhighlight>

=== Idle Timer Reset ===
Task acceptance is a key activity event that resets the auto-shutdown timer:
<syntaxhighlight lang="python">
self._reset_idle_timer()
</syntaxhighlight>

This ensures the runner stays alive while actively accepting work.

=== Rejection Reasons ===
{| class="wikitable"
|-
! Reason !! Constant !! Description
|-
| Offer expired || TASK_REJECTED_REASON_OFFER_EXPIRED || Offer no longer valid
|-
| At capacity || TASK_REJECTED_REASON_AT_CAPACITY || Max concurrent tasks reached
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Task_Acceptance]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[Implementation:n8n-io_n8n_TaskRunner_send_offers]]
* [[Implementation:n8n-io_n8n_TaskRunner_execute_task]]

=== State Transitions ===
* Offer removed from open_offers
* Task added to running_tasks with status WAITING_FOR_SETTINGS
* Idle timer reset to prevent premature shutdown

=== Message Types ===
* '''Input:''' BrokerTaskOfferAccept (offer_id, task_id)
* '''Output:''' RunnerTaskAccepted (task_id) or RunnerTaskRejected (task_id, reason)
