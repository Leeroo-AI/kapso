# KV Cache Event Monitoring Subscriber

**Source:** `examples/online_serving/kv_events_subscriber.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 117

## Overview

This example demonstrates how to monitor KV cache operations in vLLM through a ZeroMQ-based event streaming system. It provides real-time visibility into cache block lifecycle events (stored, removed, cleared) and implements a replay mechanism to handle missed messages, enabling external systems to track and react to cache state changes.

## Implementation Pattern

### Architecture Design

The implementation uses a publish-subscribe pattern with replay capability:

**vLLM Server:**
- Publishes KV cache events to ZMQ pub socket (port 5557)
- Maintains event history for replay (port 5558)
- Assigns sequential numbers to events for gap detection

**Subscriber Client:**
- Connects to pub socket for real-time events
- Detects missed messages via sequence number gaps
- Requests replays from replay socket when gaps occur
- Processes events for monitoring or cache coordination

### Use Cases

**External Cache Management:**
- Track what's cached for intelligent prefetching
- Implement cache warming strategies
- Coordinate distributed cache systems

**Debugging and Profiling:**
- Understand cache hit/miss patterns
- Identify memory pressure events
- Analyze cache effectiveness

**Resource Optimization:**
- Implement custom eviction policies
- Balance cache across multiple servers
- Predict memory requirements

## Technical Implementation

### 1. Event Type Definitions

```python
class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list[Any]


class KVCacheEvent(
    msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True
):
    """Base class for all KV cache-related events"""


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]
```

**Event Types:**

**BlockStored:**
- Fired when new KV cache blocks are stored
- Contains block hashes for identification
- Includes parent hash for prefix tracking
- Records token IDs for content inspection
- Supports LoRA-specific caching

**BlockRemoved:**
- Fired when blocks are evicted from cache
- Lists affected block hashes
- Indicates storage medium (GPU/CPU)

**AllBlocksCleared:**
- Fired when entire cache is cleared
- Used for cache resets or memory pressure responses

### 2. Message Serialization

```python
from msgspec.msgpack import Decoder

decoder = Decoder(type=KVEventBatch)
```

**Why msgspec:**
- High-performance serialization (faster than JSON or pickle)
- Type-safe deserialization with validation
- Zero-copy operations where possible
- Efficient binary encoding via MessagePack

**Struct Configuration:**
- `array_like=True`: Enables tuple-like serialization for compactness
- `omit_defaults=True`: Skips serializing default values
- `gc=False`: Disables cyclic garbage collection for performance
- `tag=True`: Adds type tags for polymorphic deserialization

### 3. Subscription Setup

```python
def main():
    decoder = Decoder(type=KVEventBatch)
    last_seq = -1

    context = zmq.Context()

    # Set up the main subscription socket
    sub = context.socket(zmq.SUB)
    sub.connect("tcp://localhost:5557")
    topic = "kv-events"
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    # Initialize replay socket
    replay = context.socket(zmq.REQ)
    replay.connect("tcp://localhost:5558")
    poller = zmq.Poller()
    poller.register(replay, zmq.POLLIN)

    print("Listening for KV cache events on topic:", topic)
```

**Socket Configuration:**

**Subscription Socket (SUB):**
- Connects to publisher at port 5557
- Subscribes to "kv-events" topic
- Receives real-time event stream

**Replay Socket (REQ):**
- Connects to replay service at port 5558
- Uses request-reply pattern
- Fetches historical events on demand

### 4. Event Processing Loop

```python
while True:
    try:
        if sub.poll(50):
            _, seq_bytes, payload = sub.recv_multipart()
            seq = int.from_bytes(seq_bytes, "big")

            if last_seq >= 0 and seq > last_seq + 1:
                missed = seq - last_seq - 1
                print(
                    f"Missed {missed} messages (last: {last_seq}, current: {seq})"
                )

                # Trigger replay mechanism
                replay.send((last_seq + 1).to_bytes(8, "big"))

                while poller.poll(timeout=200):
                    seq_bytes, replay_payload = replay.recv_multipart()
                    if not replay_payload:
                        # End of replay marker
                        break

                    replay_seq = int.from_bytes(seq_bytes, "big")

                    if replay_seq > last_seq:
                        event_batch = decoder.decode(replay_payload)
                        process_event(event_batch)
                        last_seq = replay_seq
                        if replay_seq >= seq - 1:
                            break

            event_batch = decoder.decode(payload)
            process_event(event_batch)

        # ... do other periodic work or check for shutdown ...

    except KeyboardInterrupt:
        print("Interrupted")
        break
    except Exception as e:
        print("Error decoding message:", e)
```

**Message Format:**
```
[topic, sequence_number, payload]
```

- Topic: UTF-8 string ("kv-events")
- Sequence: 8-byte big-endian integer
- Payload: msgspec-encoded KVEventBatch

### 5. Gap Detection and Replay

**Gap Detection:**
```python
if last_seq >= 0 and seq > last_seq + 1:
    missed = seq - last_seq - 1
    print(f"Missed {missed} messages")
```

Compares current sequence number with last seen sequence to detect gaps.

**Replay Request:**
```python
replay.send((last_seq + 1).to_bytes(8, "big"))
```

Sends the first missing sequence number to request replay starting from that point.

**Replay Processing:**
```python
while poller.poll(timeout=200):
    seq_bytes, replay_payload = replay.recv_multipart()
    if not replay_payload:
        # End of replay marker
        break

    replay_seq = int.from_bytes(seq_bytes, "big")

    if replay_seq > last_seq:
        event_batch = decoder.decode(replay_payload)
        process_event(event_batch)
        last_seq = replay_seq
        if replay_seq >= seq - 1:
            break
```

**Replay Protocol:**
1. Client sends starting sequence number
2. Server responds with historical events in order
3. Empty payload indicates end of replay
4. Client processes events until gap is filled

### 6. Event Processing

```python
def process_event(event_batch):
    print(f"Received event batch at {event_batch.ts}:")
    for event in event_batch.events:
        print(f"  - {event}")
```

The example provides a simple print-based processor. Production implementations would:

**For Monitoring:**
```python
def process_event(event_batch):
    for event in event_batch.events:
        if isinstance(event, BlockStored):
            metrics.increment("cache.blocks_stored", len(event.block_hashes))
            cache_tracker.add_blocks(event.block_hashes, event.token_ids)
        elif isinstance(event, BlockRemoved):
            metrics.increment("cache.blocks_removed", len(event.block_hashes))
            cache_tracker.remove_blocks(event.block_hashes)
        elif isinstance(event, AllBlocksCleared):
            metrics.increment("cache.full_clears")
            cache_tracker.clear_all()
```

**For Distributed Cache Coordination:**
```python
def process_event(event_batch):
    for event in event_batch.events:
        if isinstance(event, BlockStored):
            # Replicate to peer cache servers
            for peer in peer_caches:
                peer.store_blocks(event.block_hashes, event.token_ids)
        elif isinstance(event, BlockRemoved):
            # Sync removal across cluster
            for peer in peer_caches:
                peer.remove_blocks(event.block_hashes)
```

## Event Details

### BlockStored Event

```python
class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
```

**Fields Explained:**

**block_hashes:**
Unique identifiers for the cached blocks. A single event may store multiple consecutive blocks for efficiency.

**parent_block_hash:**
Hash of the parent block in the KV cache tree. Enables prefix sharing:
- `None` indicates root block (start of sequence)
- Non-null indicates this block extends a cached prefix

**token_ids:**
The tokens covered by these blocks. Useful for:
- Understanding cache contents
- Implementing semantic cache invalidation
- Debugging prompt processing

**block_size:**
Number of tokens per block (e.g., 16). Helps interpret token_ids array.

**lora_id:**
LoRA adapter ID if using LoRA-specific caching:
- `None` for base model cache
- Numeric ID for adapter-specific cache entries

**medium:**
Storage location (e.g., "gpu", "cpu"). Enables tracking of:
- GPU cache pressure (more CPU evictions)
- Multi-tier cache effectiveness

### BlockRemoved Event

```python
class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
```

**Eviction Scenarios:**
- LRU eviction due to memory pressure
- Manual cache invalidation
- Sequence completion and cleanup
- Cache size limit enforcement

### AllBlocksCleared Event

```python
class AllBlocksCleared(KVCacheEvent):
    pass
```

**Trigger Conditions:**
- Server restart or reconfiguration
- Out-of-memory recovery
- Admin-initiated cache clear
- Model swap or reload

## Usage Requirements

### vLLM Server Configuration

To enable KV cache event streaming, start vLLM with appropriate flags:

```bash
vllm serve model_name \
  --enable-kv-events \
  --kv-events-pub-port 5557 \
  --kv-events-replay-port 5558
```

**Configuration Options:**
- `--enable-kv-events`: Activates event publishing
- `--kv-events-pub-port`: Port for real-time event stream (default: 5557)
- `--kv-events-replay-port`: Port for replay service (default: 5558)
- `--kv-events-topic`: Custom topic name (default: "kv-events")

### Dependencies

```python
# Event serialization
import msgspec
from msgspec.msgpack import Decoder

# Networking
import zmq

# vLLM types
from vllm.v1.core.kv_cache_utils import ExternalBlockHash
```

**Installation:**
```bash
pip install msgspec pyzmq vllm
```

### Network Considerations

**Firewall Rules:**
Open ports 5557 and 5558 for subscriber access.

**Local vs. Remote:**
- Example uses `localhost` for same-machine monitoring
- Change to server IP for remote monitoring: `tcp://vllm-server:5557`

**Multiple Subscribers:**
ZMQ PUB-SUB supports multiple subscribers without configuration changes.

## Production Patterns

### Reliable Event Processing

```python
class ReliableEventSubscriber:
    def __init__(self, server_address, topic="kv-events"):
        self.server_address = server_address
        self.topic = topic
        self.decoder = Decoder(type=KVEventBatch)
        self.last_seq = -1
        self.setup_sockets()

    def setup_sockets(self):
        self.context = zmq.Context()

        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect(f"tcp://{self.server_address}:5557")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.sub.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

        self.replay = self.context.socket(zmq.REQ)
        self.replay.connect(f"tcp://{self.server_address}:5558")
        self.replay.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout

    def run(self):
        while not self.should_stop:
            try:
                self.process_messages()
            except zmq.Again:
                # Timeout, continue
                continue
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                self.reconnect()

    def reconnect(self):
        logger.info("Reconnecting...")
        self.context.term()
        time.sleep(5)
        self.setup_sockets()
```

### State Tracking

```python
class CacheStateTracker:
    def __init__(self):
        self.blocks = {}  # block_hash -> (tokens, lora_id)
        self.token_index = defaultdict(set)  # token -> {block_hashes}

    def handle_event(self, event):
        if isinstance(event, BlockStored):
            for block_hash in event.block_hashes:
                self.blocks[block_hash] = (event.token_ids, event.lora_id)
                for token in event.token_ids:
                    self.token_index[token].add(block_hash)

        elif isinstance(event, BlockRemoved):
            for block_hash in event.block_hashes:
                if block_hash in self.blocks:
                    tokens, _ = self.blocks[block_hash]
                    for token in tokens:
                        self.token_index[token].discard(block_hash)
                    del self.blocks[block_hash]

        elif isinstance(event, AllBlocksCleared):
            self.blocks.clear()
            self.token_index.clear()

    def query_cached_tokens(self, tokens):
        """Check if a token sequence is cached."""
        if not tokens:
            return False

        # Find blocks containing first token
        candidates = self.token_index.get(tokens[0], set())

        for block_hash in candidates:
            cached_tokens, _ = self.blocks[block_hash]
            if tokens == cached_tokens[:len(tokens)]:
                return True

        return False
```

### Metrics Collection

```python
class CacheMetricsCollector:
    def __init__(self):
        self.metrics = {
            "blocks_stored": 0,
            "blocks_removed": 0,
            "cache_clears": 0,
            "gpu_blocks": 0,
            "cpu_blocks": 0,
        }

    def process_event(self, event_batch):
        for event in event_batch.events:
            if isinstance(event, BlockStored):
                self.metrics["blocks_stored"] += len(event.block_hashes)
                if event.medium == "gpu":
                    self.metrics["gpu_blocks"] += len(event.block_hashes)
                elif event.medium == "cpu":
                    self.metrics["cpu_blocks"] += len(event.block_hashes)

            elif isinstance(event, BlockRemoved):
                self.metrics["blocks_removed"] += len(event.block_hashes)
                if event.medium == "gpu":
                    self.metrics["gpu_blocks"] -= len(event.block_hashes)
                elif event.medium == "cpu":
                    self.metrics["cpu_blocks"] -= len(event.block_hashes)

            elif isinstance(event, AllBlocksCleared):
                self.metrics["cache_clears"] += 1
                self.metrics["gpu_blocks"] = 0
                self.metrics["cpu_blocks"] = 0

    def get_metrics(self):
        return self.metrics.copy()
```

## Performance Considerations

### Message Volume

**Typical Rates:**
- Low traffic: 10-100 events/second
- Medium traffic: 100-1000 events/second
- High traffic: 1000+ events/second

**Batching:**
Events are sent in batches to reduce overhead. Batch size depends on:
- Event generation rate
- Configured batch interval
- Network conditions

### Subscriber Performance

**Processing Time:**
Keep `process_event()` fast to avoid falling behind:
- Use async processing for I/O operations
- Buffer events for batch processing
- Offload heavy computation to worker threads

**Backpressure:**
If subscriber can't keep up:
- Gaps will occur more frequently
- Replay will be triggered repeatedly
- Consider multiple subscribers with load balancing

### Network Bandwidth

**Typical Event Sizes:**
- BlockStored: 100-500 bytes (depends on block size)
- BlockRemoved: 50-200 bytes
- AllBlocksCleared: ~20 bytes

**Bandwidth Estimation:**
At 1000 events/sec with avg 200 bytes:
- 200 KB/sec = 1.6 Mbps
- Negligible for modern networks

## Troubleshooting

### Common Issues

**Missing Events:**
```
Missed 10 messages (last: 42, current: 53)
```

**Causes:**
- Subscriber processing too slow
- Network congestion
- ZMQ buffer overflow

**Solutions:**
- Increase ZMQ high water mark: `sub.setsockopt(zmq.RCVHWM, 10000)`
- Optimize event processing
- Add multiple subscribers with partitioning

**Replay Timeouts:**
```
Error decoding message: zmq.Again
```

**Causes:**
- Replay service overloaded
- Network issues
- Server not maintaining history

**Solutions:**
- Increase replay timeout
- Reduce replay request rate
- Check server configuration

**Decoding Errors:**
```
Error decoding message: ValidationError
```

**Causes:**
- Version mismatch between server and client
- Corrupted messages
- Schema changes

**Solutions:**
- Ensure matching vLLM versions
- Update event type definitions
- Add version checking

## Related Systems

### External Cache Implementations

The event system enables building:

**Distributed Cache:**
Replicate cache state across multiple vLLM instances for consistent prefix reuse.

**Persistent Cache:**
Store frequently-used blocks to disk/database for cache warming on restart.

**Semantic Cache:**
Index cache by content for intelligent invalidation when data changes.

### Integration Examples

**Prometheus Metrics:**
Export cache metrics for monitoring and alerting.

**OpenTelemetry Tracing:**
Correlate cache events with request traces for performance analysis.

**Custom Load Balancers:**
Route requests to instances with cached prefixes for optimal performance.

## References

- **ZeroMQ PUB-SUB:** [ZMQ Guide](https://zguide.zeromq.org/docs/chapter2/)
- **msgspec:** [Documentation](https://jcristharif.com/msgspec/)
- **vLLM KV Cache:** Internal cache implementation details
- **Replay Pattern:** Ensures reliable event delivery in distributed systems
