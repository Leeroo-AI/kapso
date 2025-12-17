{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]], [[domain::Load_Balancing]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Pull-based task distribution where workers proactively advertise their available capacity to a central broker, enabling demand-driven load balancing.

=== Description ===

Offer-based distribution inverts the traditional push model of task assignment. Instead of a central coordinator deciding which worker should receive a task, workers themselves signal their availability and capacity by sending "offers" to execute tasks. The broker acts as a matchmaker, accepting offers when tasks are available and workers have capacity.

This approach solves several key problems in distributed task execution:
* **Overload Prevention**: Workers control their own load by limiting offers
* **Backpressure**: Natural flow control as workers stop offering when saturated
* **Worker Autonomy**: Each worker manages its own capacity and resource limits
* **Scalability**: Broker complexity remains constant regardless of worker count

The pattern is particularly effective when workers have heterogeneous capabilities, varying load levels, or dynamic capacity constraints.

=== Usage ===

Apply this principle when designing distributed systems where:
* Workers have varying capacity or capability profiles
* Preventing worker overload is critical to system stability
* Tasks arrive at variable rates requiring dynamic load balancing
* Workers need autonomy to manage their own resources
* System must gracefully handle worker failures or slowdowns
* Natural backpressure mechanism is preferred over queue depth limits

== Theoretical Basis ==

The offer-based model implements a **pull-based queue** where consumers drive demand:

**Traditional Push Model:**
```
Broker has task → Select worker → Push task to worker
Problem: Broker must track worker capacity and may overload workers
```

**Offer-Based Pull Model:**
```
Worker has capacity → Send offer to broker → Broker accepts if task available
Advantage: Worker controls admission, natural backpressure
```

Key algorithmic properties:

1. **Capacity Advertisement**: Each worker sends offers up to its concurrency limit
   ```
   offers_to_send = max_concurrency - active_tasks
   ```

2. **Offer Expiration**: Offers include TTL to prevent stale assignments
   ```
   if current_time > offer_timestamp + offer_ttl:
       reject_offer()
   ```

3. **Admission Control**: Worker validates capacity before accepting
   ```
   if active_tasks >= max_concurrency:
       reject_task()
   ```

4. **Backpressure Propagation**: When workers stop offering, broker naturally queues tasks

This design provides **work-conserving scheduling** where tasks execute immediately when capacity exists, while maintaining **stable load** that never exceeds worker limits.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_send_offers]]

=== Related Principles ===
* [[Principle:n8n-io_n8n_WebSocket_Connection]] - Transport layer for offer messages
* [[Principle:n8n-io_n8n_Task_Acceptance]] - Validation logic when offers are accepted
* [[Principle:n8n-io_n8n_Task_Completion]] - Frees capacity for new offers
