{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]], [[domain::Real_Time_Communication]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Persistent WebSocket communication establishes a long-lived bidirectional connection between task runners and brokers for real-time task coordination.

=== Description ===

WebSocket connections provide a persistent, full-duplex communication channel that enables low-latency message exchange between task runners and the central broker. Unlike traditional request-response patterns, the WebSocket remains open throughout the task runner's lifetime, allowing the broker to push task assignments immediately and the runner to send status updates without polling overhead.

This architecture solves the problem of efficient task coordination in distributed systems where tasks arrive asynchronously and workers need to signal availability in real-time. The persistent connection eliminates the latency and overhead of repeatedly establishing HTTP connections for each interaction.

=== Usage ===

Apply this principle when building distributed task execution systems where:
* Workers need to receive task assignments with minimal latency
* Workers must send real-time status updates (offers, completions, errors)
* The coordinator needs to know worker availability without polling
* Task distribution requires bidirectional communication
* System benefits from push-based notifications rather than pull-based polling

== Theoretical Basis ==

The WebSocket protocol upgrades from HTTP to provide persistent TCP connections with message framing:

1. **Connection Establishment**: Initial HTTP handshake upgrades to WebSocket protocol
2. **Message Exchange**: Bidirectional frame-based message passing over single TCP connection
3. **Keep-Alive**: Periodic ping/pong frames maintain connection liveness
4. **Graceful Closure**: Clean shutdown protocol with close frames

Key advantages over HTTP polling:
* Eliminates connection setup overhead (TCP handshake, TLS negotiation)
* Reduces latency from request-response cycles to single message frame
* Decreases bandwidth usage by removing HTTP headers on each request
* Enables server-initiated communication without client polling

Message flow pattern:
```
Runner → Broker: task:offer (worker advertises capacity)
Broker → Runner: task:offer:accept (broker assigns task)
Runner → Broker: task:done (worker reports completion)
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_start]]

=== Related Principles ===
* [[Principle:n8n-io_n8n_Offer_Based_Distribution]] - Uses WebSocket for offer messages
* [[Principle:n8n-io_n8n_Task_Acceptance]] - Uses WebSocket for acceptance protocol
* [[Principle:n8n-io_n8n_Task_Completion]] - Uses WebSocket for completion notifications
