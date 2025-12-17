{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]], [[domain::Concurrency_Control]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Protocol for validating and accepting task assignments ensures workers only commit to tasks they can execute within capacity and time constraints.

=== Description ===

Task acceptance implements a multi-stage validation protocol that verifies both temporal and resource constraints before a worker commits to executing a task. When a broker accepts a worker's offer and assigns a task, the worker must validate that:
1. The original offer hasn't expired (time validity)
2. The worker still has available capacity (resource validity)
3. The task state can be successfully created (initialization validity)

This protocol solves the race condition problem inherent in distributed systems: the time between a worker sending an offer and receiving an acceptance may result in changed conditions. Without validation, workers could accept tasks they cannot execute, leading to failures, timeouts, or system instability.

The acceptance phase acts as a critical synchronization point, ensuring consistency between the broker's view of worker capacity and the worker's actual state.

=== Usage ===

Apply this principle when designing task distribution systems where:
* Time elapses between capacity advertisement and task assignment
* Workers may receive multiple simultaneous task assignments
* Resource availability can change between offer and acceptance
* System must guarantee workers don't exceed capacity limits
* Offer expiration provides time-bounded validity windows
* Failed acceptances should trigger graceful fallback (offer to another worker)

== Theoretical Basis ==

Task acceptance implements a **two-phase commit** pattern for capacity reservation:

**Phase 1: Offer (Tentative Reservation)**
```
Worker: "I have capacity for N tasks"
Broker: Receives offer, stores with timestamp
```

**Phase 2: Accept (Confirmed Reservation)**
```
Broker: "I accept your offer for task T"
Worker: Validate offer_id, check expiration, verify capacity
If valid: Create task state, execute
If invalid: Reject and broker reassigns
```

Key validation checks:

1. **Temporal Validity**:
   ```
   if offer_timestamp + TTL < current_time:
       return OFFER_EXPIRED
   ```

2. **Capacity Validity**:
   ```
   if active_tasks >= max_concurrency:
       return CAPACITY_EXCEEDED
   ```

3. **Idempotency Check**:
   ```
   if task_id in active_tasks:
       return DUPLICATE_ASSIGNMENT
   ```

This protocol provides **at-most-once execution semantics**: a task is only executed if validation succeeds, preventing double execution or overload conditions.

**Race Condition Handling**:
```
T0: Worker sends offer (capacity = 1)
T1: Broker sends task A
T2: Broker sends task B (before receiving A acceptance)
T3: Worker accepts A, capacity = 0
T4: Worker rejects B (capacity exceeded)
```

The validation at T4 prevents overload, and the broker can reassign task B to another worker.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_handle_task_offer_accept]]

=== Related Principles ===
* [[Principle:n8n-io_n8n_Offer_Based_Distribution]] - Generates offers that are later accepted
* [[Principle:n8n-io_n8n_WebSocket_Connection]] - Transport for acceptance messages
* [[Principle:n8n-io_n8n_Task_Completion]] - Follows successful acceptance
