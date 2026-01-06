# Heuristic: Offer Validity Jitter

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Code|task_runner.py|packages/@n8n/task-runner-python/src/task_runner.py]]
* [[source::Code|constants.py|packages/@n8n/task-runner-python/src/constants.py]]
|-
! Domains
| [[domain::Distributed_Systems]], [[domain::Load_Balancing]], [[domain::Reliability]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Add random jitter (0-500ms) to task offer validity periods to prevent thundering herd effects when multiple runners offer capacity simultaneously.

=== Description ===
In distributed task execution, multiple Python Task Runners may offer capacity to the task broker at similar times. If all offers expire at exactly the same moment, a "thundering herd" effect can occur where all runners simultaneously re-offer, creating load spikes on the broker. Adding random jitter to offer validity spreads out expiration times, smoothing the load.

=== Usage ===
This heuristic is automatically applied in `TaskRunner._send_offers()`. No configuration is needed. The jitter is transparent to users but improves system stability.

== The Insight (Rule of Thumb) ==

* **Action:** Add random jitter to offer validity: `valid_for_ms = base + random(0, max_jitter)`
* **Value:**
  * `OFFER_VALIDITY = 5000ms` (base validity)
  * `OFFER_VALIDITY_MAX_JITTER = 500ms` (max random addition)
* **Trade-off:** Offers valid for 5000-5500ms instead of exactly 5000ms
* **Benefit:** Prevents synchronized expiration across multiple runners

== Reasoning ==

1. **Thundering herd problem:** Without jitter, N runners with identical validity periods create N simultaneous re-offer events

2. **Random distribution:** With 500ms jitter across N runners, expirations spread over 500ms window

3. **Minimal impact:** 500ms variance on a 5s validity period is only 10% - imperceptible to users

4. **Broker protection:** Smooths load on task broker, especially in scaled deployments

5. **Latency buffer:** Combined with `OFFER_VALIDITY_LATENCY_BUFFER = 0.1s`, accounts for network delays

== Code Evidence ==

From `constants.py:33-36`:

<syntaxhighlight lang="python">
OFFER_INTERVAL = 0.25  # 250ms
OFFER_VALIDITY = 5000  # ms
OFFER_VALIDITY_MAX_JITTER = 500  # ms
OFFER_VALIDITY_LATENCY_BUFFER = 0.1  # 100ms
</syntaxhighlight>

From `task_runner.py:461`:

<syntaxhighlight lang="python">
valid_for_ms = OFFER_VALIDITY + random.randint(0, OFFER_VALIDITY_MAX_JITTER)
</syntaxhighlight>

Full context from `task_runner.py:454-473`:

<syntaxhighlight lang="python">
offers_to_send = self.config.max_concurrency - (
    self.running_tasks_count + len(self.open_offers)
)

for _ in range(offers_to_send):
    offer_id = generate_id()
    valid_for_ms = OFFER_VALIDITY + random.randint(0, OFFER_VALIDITY_MAX_JITTER)
    valid_until = time.time() + (valid_for_ms / 1000) - OFFER_VALIDITY_LATENCY_BUFFER

    self.open_offers[offer_id] = TaskOffer(
        offer_id=offer_id,
        valid_until=valid_until,
    )

    await self._send(
        RunnerTaskOffer(
            type="runner:taskoffer",
            taskType=TASK_TYPE_PYTHON,
            offerId=offer_id,
            validFor=valid_for_ms,
        )
    )
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_TaskRunner_send_offers]]
* [[uses_heuristic::Workflow:n8n-io_n8n_Python_Task_Execution]]
* [[uses_heuristic::Principle:n8n-io_n8n_Task_Offer_Negotiation]]
