# Principle: Broker Connection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|WebSocket Protocol|https://datatracker.ietf.org/doc/html/rfc6455]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]], [[domain::Network_Communication]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for establishing and maintaining a persistent WebSocket connection between a task runner and a central broker for bidirectional task communication.

=== Description ===

Broker Connection establishes the communication channel between distributed task runners and a centralized task broker. This enables:

1. **Authentication**: Runners authenticate with the broker using a grant token
2. **Persistent Connection**: WebSocket provides full-duplex, persistent communication
3. **Message Routing**: The broker routes tasks to available runners based on capacity
4. **Reconnection Handling**: Automatic retry logic handles transient network failures

The broker-runner architecture decouples task producers from executors, enabling horizontal scaling of execution capacity.

=== Usage ===

Apply this principle when:
- Building distributed task execution systems
- Implementing worker nodes that need persistent communication with a coordinator
- Designing systems requiring real-time bidirectional messaging
- Creating fault-tolerant distributed architectures

== Theoretical Basis ==

The connection follows a **Persistent Worker** pattern with automatic reconnection:

<syntaxhighlight lang="python">
# Pseudo-code for broker connection
async def connect_to_broker():
    while not shutting_down:
        try:
            # 1. Establish authenticated WebSocket connection
            connection = await websocket.connect(
                broker_url,
                headers={"Authorization": f"Bearer {token}"}
            )

            # 2. Listen for messages until disconnection
            await listen_for_messages(connection)

        except ConnectionError:
            # 3. Reconnect with backoff
            await sleep(reconnect_delay)
</syntaxhighlight>

The WebSocket protocol ensures:
- Low latency (no HTTP overhead per message)
- Server push capability (broker can send tasks without polling)
- Connection state awareness (detect disconnections immediately)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_start]]
