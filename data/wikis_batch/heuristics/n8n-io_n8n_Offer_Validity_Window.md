# Heuristic: Offer Validity Window

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n Task Runner Python|https://github.com/n8n-io/n8n/tree/master/packages/@n8n/task-runner-python]]
|-
! Domains
| [[domain::Distributed_Systems]], [[domain::Load_Balancing]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Task offers expire after 5000ms plus random jitter (0-500ms) with 100ms latency buffer, preventing stale task assignments in the pull-based distribution model.

=== Description ===

The n8n Python Task Runner uses an offer-based (pull) model for task distribution. Runners periodically send "offers" to the broker indicating their capacity to accept tasks. Each offer has a validity window after which it expires. This prevents the broker from assigning tasks to offers that may no longer reflect the runner's current state.

=== Usage ===

This heuristic applies automatically to all task offer generation in **TaskRunner._send_offers()** and **TaskRunner._send_offers_loop()**. Understanding this timing is critical when:

- Debugging task rejection with "Offer expired" messages
- Tuning for high-latency network environments
- Understanding why tasks are rejected despite available capacity

== The Insight (Rule of Thumb) ==

* **Base Validity:** `OFFER_VALIDITY = 5000` ms (5 seconds)
* **Jitter Range:** `OFFER_VALIDITY_MAX_JITTER = 500` ms (0-500ms random)
* **Latency Buffer:** `OFFER_VALIDITY_LATENCY_BUFFER = 0.1` (100ms subtracted from validity for network latency)
* **Offer Interval:** `OFFER_INTERVAL = 0.25` (250ms between offer cycles)
* **Trade-off:** Shorter windows reduce stale assignments but increase offer traffic; longer windows reduce traffic but risk stale assignments

== Reasoning ==

The offer validity timing parameters are designed to handle distributed system challenges:

1. **5-Second Base Window:** Long enough for the broker to process and respond to offers, short enough to reflect current runner state
2. **Random Jitter (0-500ms):** Prevents thundering herd when multiple runners start simultaneously
3. **100ms Latency Buffer:** Account for network round-trip time so runner rejects before broker considers expired
4. **250ms Offer Interval:** Balance between responsiveness and reducing unnecessary network traffic

The formula for effective offer expiry:
```
effective_validity = base_validity + random(0, jitter) - latency_buffer
                   = 5000 + random(0, 500) - 100
                   = 4900 to 5400 ms
```

== Code Evidence ==

Constants from `constants.py:33-36`:
<syntaxhighlight lang="python">
OFFER_INTERVAL = 0.25  # 250ms
OFFER_VALIDITY = 5000  # ms
OFFER_VALIDITY_MAX_JITTER = 500  # ms
OFFER_VALIDITY_LATENCY_BUFFER = 0.1  # 100ms
</syntaxhighlight>

Rejection reason from `constants.py:119-120`:
<syntaxhighlight lang="python">
TASK_REJECTED_REASON_OFFER_EXPIRED = (
    "Offer expired - not accepted within validity window"
)
</syntaxhighlight>

Capacity rejection from `constants.py:122`:
<syntaxhighlight lang="python">
TASK_REJECTED_REASON_AT_CAPACITY = "No open task slots - runner already at capacity"
</syntaxhighlight>

Offer generation in `task_runner.py:462-473` (inferred from Implementation page):
<syntaxhighlight lang="python">
async def _send_offers(self):
    """Send task offers based on current capacity."""
    available_slots = self._config.max_concurrency - self.running_tasks_count
    offers_needed = available_slots - len(self._open_offers)

    for _ in range(offers_needed):
        jitter = random.uniform(0, OFFER_VALIDITY_MAX_JITTER)
        validity = OFFER_VALIDITY + jitter

        offer = RunnerTaskOffer(
            offer_id=generate_nanoid(),
            valid_until=time.time() + (validity / 1000)
        )
        self._open_offers[offer.offer_id] = offer
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_TaskRunner_send_offers]]
* [[uses_heuristic::Implementation:n8n-io_n8n_TaskRunner_handle_task_offer_accept]]
* [[uses_heuristic::Principle:n8n-io_n8n_Offer_Based_Distribution]]
* [[uses_heuristic::Principle:n8n-io_n8n_Task_Acceptance]]
