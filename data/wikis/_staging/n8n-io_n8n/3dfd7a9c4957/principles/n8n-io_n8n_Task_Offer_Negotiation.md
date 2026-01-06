# Principle: Task Offer Negotiation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]], [[domain::Resource_Management]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for implementing a capacity-aware task distribution protocol where runners advertise available slots and brokers match tasks to capacity.

=== Description ===

Task Offer Negotiation implements a pull-based work distribution pattern:

1. **Capacity Advertisement**: Runners periodically send "offers" indicating they can accept work
2. **Offer Validity**: Each offer has a time-limited validity window with jitter to prevent thundering herd
3. **Acceptance Handling**: When the broker accepts an offer, the runner validates it hasn't expired
4. **Rejection Logic**: Runners reject tasks if at capacity or if the offer has expired

This pattern enables:
- **Load Balancing**: Work flows naturally to runners with available capacity
- **Backpressure**: Runners only receive work they can handle
- **Fault Tolerance**: Expired offers prevent stale task assignments

=== Usage ===

Apply this principle when:
- Implementing distributed work queues with heterogeneous workers
- Building systems where workers have varying capacity or capability
- Designing pull-based task distribution (vs. push-based assignment)
- Creating backpressure-aware distributed systems

== Theoretical Basis ==

The negotiation follows an **Offer-Accept** protocol:

<syntaxhighlight lang="python">
# Pseudo-code for offer negotiation

# Runner side: periodically send offers
async def send_offers():
    while can_send_offers:
        # Calculate available slots
        available = max_concurrency - running_tasks - open_offers

        for _ in range(available):
            offer_id = generate_id()
            valid_until = now() + VALIDITY + random_jitter()

            open_offers[offer_id] = Offer(offer_id, valid_until)
            send(RunnerTaskOffer(offer_id, valid_for=VALIDITY))

        await sleep(OFFER_INTERVAL)

# Runner side: handle broker acceptance
async def handle_accept(offer_id, task_id):
    offer = open_offers.get(offer_id)

    if offer is None or offer.has_expired:
        send(TaskRejected(task_id, "offer_expired"))
        return

    if at_capacity:
        send(TaskRejected(task_id, "at_capacity"))
        return

    del open_offers[offer_id]
    running_tasks[task_id] = TaskState(task_id)
    send(TaskAccepted(task_id))
</syntaxhighlight>

Key timing parameters:
- **OFFER_INTERVAL**: 0.25s between offer batches
- **OFFER_VALIDITY**: 5000ms base validity
- **OFFER_VALIDITY_MAX_JITTER**: Random jitter to prevent synchronization
- **OFFER_VALIDITY_LATENCY_BUFFER**: Buffer for network latency

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_send_offers]]
