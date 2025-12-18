# Implementation: TaskRunner._send_offers

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete methods for managing task offer lifecycle including periodic offer sending and acceptance handling.

=== Description ===

The offer negotiation implementation consists of two main methods:

1. **`_send_offers()`**: Sends capacity-based task offers to the broker
   - Cleans up expired offers
   - Calculates available slots based on concurrency, running tasks, and open offers
   - Creates offers with jittered validity periods
   - Sends `RunnerTaskOffer` messages to broker

2. **`_handle_task_offer_accept()`**: Processes broker's offer acceptance
   - Validates offer exists and hasn't expired
   - Checks runner isn't at capacity
   - Registers task in `running_tasks` dictionary
   - Sends acceptance/rejection response

=== Usage ===

These methods are called automatically by the TaskRunner. `_send_offers_loop()` runs continuously after registration, and `_handle_task_offer_accept()` is triggered by broker messages.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L441-474 (_send_offers), L253-280 (_handle_task_offer_accept)

=== Signature ===
<syntaxhighlight lang="python">
async def _send_offers(self) -> None:
    """
    Send task offers based on available capacity.

    Cleans up expired offers before calculating how many new
    offers to send based on max_concurrency minus current load.
    """

async def _handle_task_offer_accept(self, message: BrokerTaskOfferAccept) -> None:
    """
    Handle broker's acceptance of a task offer.

    Args:
        message: Contains offer_id and task_id from broker.

    Sends:
        RunnerTaskAccepted: If offer valid and capacity available.
        RunnerTaskRejected: If offer expired or at capacity.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal methods - accessed via TaskRunner instance
from src.task_runner import TaskRunner
</syntaxhighlight>

== I/O Contract ==

=== _send_offers Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (uses self) || - || - || Uses self.config.max_concurrency, self.open_offers, self.running_tasks
|}

=== _send_offers Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| self.open_offers || dict[str, TaskOffer] || Updated with new offers, expired offers removed
|-
| (WebSocket) || RunnerTaskOffer || Offer messages sent to broker
|}

=== _handle_task_offer_accept Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| message || BrokerTaskOfferAccept || Yes || Contains offer_id and task_id
|}

=== _handle_task_offer_accept Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| self.running_tasks || dict[str, TaskState] || New task added if accepted
|-
| self.open_offers || dict[str, TaskOffer] || Accepted offer removed
|-
| (WebSocket) || RunnerTaskAccepted/Rejected || Response sent to broker
|}

== Usage Examples ==

=== Internal Offer Loop ===
<syntaxhighlight lang="python">
# This runs automatically after registration
# Shown for understanding the internal flow

async def _send_offers_loop(self) -> None:
    while self.can_send_offers:
        try:
            await self._send_offers()
            await asyncio.sleep(OFFER_INTERVAL)  # 0.25s
        except asyncio.CancelledError:
            break
        except Exception as e:
            self.logger.error(f"Error sending offers: {e}")
</syntaxhighlight>

=== TaskOffer Dataclass ===
<syntaxhighlight lang="python">
@dataclass
class TaskOffer:
    offer_id: str
    valid_until: float  # Unix timestamp

    @property
    def has_expired(self) -> bool:
        return time.time() > self.valid_until
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Task_Offer_Negotiation]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
