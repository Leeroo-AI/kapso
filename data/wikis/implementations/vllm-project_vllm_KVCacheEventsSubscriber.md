{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Online Serving]], [[domain::KV Cache]], [[domain::Monitoring]], [[domain::ZMQ]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates how to subscribe to KV cache events from vLLM using ZeroMQ, enabling real-time monitoring of cache block operations for external cache management systems.

=== Description ===
This example shows how to build an external client that subscribes to KV cache events published by vLLM's v1 engine via ZeroMQ. The client receives real-time notifications about cache block storage, removal, and clearing operations, along with associated metadata like block hashes, token IDs, and LoRA identifiers.

The subscriber implements a robust pattern with:
* Sequence number tracking to detect missed messages
* Automatic replay mechanism to recover missed events
* Message deserialization using msgspec for efficient parsing
* Continuous monitoring loop with proper error handling

This enables building external KV cache management systems, persistent caches, distributed cache coordination, or monitoring tools that need visibility into vLLM's cache operations.

=== Usage ===
Use this approach when:
* Building distributed KV cache systems across multiple vLLM instances
* Implementing persistent cache storage for long-term context retention
* Creating monitoring dashboards for cache utilization and performance
* Developing external cache eviction policies or optimization strategies
* Coordinating cache state between vLLM and external storage systems
* Debugging cache behavior or analyzing cache hit/miss patterns
* Implementing prefix caching across service restarts

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/kv_events_subscriber.py examples/online_serving/kv_events_subscriber.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server with KV event publishing enabled
vllm serve meta-llama/Llama-3.2-1B \
    --kv-cache-events-pub-addr tcp://localhost:5557 \
    --kv-cache-events-replay-addr tcp://localhost:5558

# In another terminal, run the subscriber
python examples/online_serving/kv_events_subscriber.py

# The subscriber will continuously print cache events
# Press Ctrl+C to stop
</syntaxhighlight>

== Key Concepts ==

=== KV Cache Events ===
vLLM's v1 engine publishes three types of cache events:

'''BlockStored:''' Notifies when cache blocks are stored
* <code>block_hashes</code>: List of block hash identifiers
* <code>parent_block_hash</code>: Parent block for prefix sharing
* <code>token_ids</code>: Tokens contained in the block
* <code>block_size</code>: Number of positions per block
* <code>lora_id</code>: Associated LoRA adapter ID (if any)
* <code>medium</code>: Storage medium (GPU, CPU, etc.)

'''BlockRemoved:''' Notifies when cache blocks are evicted
* <code>block_hashes</code>: List of removed block hashes
* <code>medium</code>: Storage medium the blocks were removed from

'''AllBlocksCleared:''' Notifies when entire cache is cleared
* No additional fields (complete cache reset)

=== ZeroMQ PUB-SUB Pattern ===
The example uses ZeroMQ's publish-subscribe pattern:

'''Publisher (vLLM):'''
* Publishes events to TCP socket (default: 5557)
* Includes sequence numbers for ordering
* No delivery guarantees (fire-and-forget)

'''Subscriber (client):'''
* Connects to publisher socket
* Filters messages by topic ("kv-events")
* Receives events asynchronously
* Must handle missed messages

=== Sequence Tracking and Replay ===
The subscriber implements reliability mechanisms:

'''Sequence Number Tracking:'''
<syntaxhighlight lang="python">
last_seq = -1

_, seq_bytes, payload = sub.recv_multipart()
seq = int.from_bytes(seq_bytes, "big")

if last_seq >= 0 and seq > last_seq + 1:
    # Detected gap - trigger replay
    missed = seq - last_seq - 1
</syntaxhighlight>

'''Replay Mechanism:'''
* Connects to separate replay socket (default: 5558)
* Requests missed messages by sequence number
* Receives historical events to fill gaps
* Continues to live stream after catching up

=== Message Format ===
Events are serialized using msgspec with msgpack encoding:
* Binary format for efficiency
* Type-safe deserialization
* Structural typing with tagged unions
* Zero-copy deserialization where possible

== Usage Examples ==

=== Basic Event Subscriber ===
<syntaxhighlight lang="python">
import zmq
from msgspec.msgpack import Decoder

# Define event types (copied from vllm.distributed.kv_events)
from vllm.v1.core.kv_cache_utils import ExternalBlockHash

class KVCacheEvent:
    pass

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

class KVEventBatch:
    ts: float
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]

# Set up subscriber
context = zmq.Context()
sub = context.socket(zmq.SUB)
sub.connect("tcp://localhost:5557")
sub.setsockopt_string(zmq.SUBSCRIBE, "kv-events")

decoder = Decoder(type=KVEventBatch)

# Process events
while True:
    try:
        if sub.poll(50):  # Poll with 50ms timeout
            _, seq_bytes, payload = sub.recv_multipart()
            seq = int.from_bytes(seq_bytes, "big")

            event_batch = decoder.decode(payload)
            print(f"Received batch at {event_batch.ts}:")
            for event in event_batch.events:
                print(f"  {event}")
    except KeyboardInterrupt:
        break
</syntaxhighlight>

=== Cache Statistics Tracking ===
<syntaxhighlight lang="python">
class CacheStats:
    def __init__(self):
        self.blocks_stored = 0
        self.blocks_removed = 0
        self.total_clears = 0
        self.active_blocks = set()

    def process_event(self, event):
        if isinstance(event, BlockStored):
            self.blocks_stored += len(event.block_hashes)
            self.active_blocks.update(event.block_hashes)
            print(f"Stored {len(event.block_hashes)} blocks, "
                  f"active: {len(self.active_blocks)}")

        elif isinstance(event, BlockRemoved):
            self.blocks_removed += len(event.block_hashes)
            self.active_blocks.difference_update(event.block_hashes)
            print(f"Removed {len(event.block_hashes)} blocks, "
                  f"active: {len(self.active_blocks)}")

        elif isinstance(event, AllBlocksCleared):
            self.total_clears += 1
            self.active_blocks.clear()
            print(f"Cache cleared (total clears: {self.total_clears})")

stats = CacheStats()

while True:
    if sub.poll(50):
        _, seq_bytes, payload = sub.recv_multipart()
        event_batch = decoder.decode(payload)

        for event in event_batch.events:
            stats.process_event(event)
</syntaxhighlight>

=== Persistent Cache Backup ===
<syntaxhighlight lang="python">
import sqlite3

# Create persistent cache database
conn = sqlite3.connect("kv_cache_backup.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS cache_blocks (
        block_hash TEXT PRIMARY KEY,
        parent_hash TEXT,
        token_ids BLOB,
        block_size INTEGER,
        lora_id INTEGER,
        medium TEXT,
        timestamp REAL
    )
""")
conn.commit()

def backup_block_stored(event, timestamp):
    for block_hash in event.block_hashes:
        cursor.execute("""
            INSERT OR REPLACE INTO cache_blocks
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(block_hash),
            str(event.parent_block_hash) if event.parent_block_hash else None,
            bytes(event.token_ids),
            event.block_size,
            event.lora_id,
            event.medium,
            timestamp
        ))
    conn.commit()

def remove_blocks(event):
    for block_hash in event.block_hashes:
        cursor.execute("DELETE FROM cache_blocks WHERE block_hash = ?",
                      (str(block_hash),))
    conn.commit()

# Subscribe and backup
while True:
    if sub.poll(50):
        _, seq_bytes, payload = sub.recv_multipart()
        event_batch = decoder.decode(payload)

        for event in event_batch.events:
            if isinstance(event, BlockStored):
                backup_block_stored(event, event_batch.ts)
            elif isinstance(event, BlockRemoved):
                remove_blocks(event)
            elif isinstance(event, AllBlocksCleared):
                cursor.execute("DELETE FROM cache_blocks")
                conn.commit()
</syntaxhighlight>

=== Multi-Instance Cache Coordination ===
<syntaxhighlight lang="python">
# Coordinate KV cache across multiple vLLM instances
import zmq

# Subscribe to multiple vLLM instances
instances = [
    ("instance-1", "tcp://node1:5557"),
    ("instance-2", "tcp://node2:5557"),
]

context = zmq.Context()
poller = zmq.Poller()

subscribers = {}
for name, addr in instances:
    sub = context.socket(zmq.SUB)
    sub.connect(addr)
    sub.setsockopt_string(zmq.SUBSCRIBE, "kv-events")
    subscribers[name] = sub
    poller.register(sub, zmq.POLLIN)

# Track which blocks are available where
block_locations = {}  # block_hash -> set of instance names

decoder = Decoder(type=KVEventBatch)

while True:
    socks = dict(poller.poll(50))

    for name, sub in subscribers.items():
        if sub in socks:
            _, seq_bytes, payload = sub.recv_multipart()
            event_batch = decoder.decode(payload)

            for event in event_batch.events:
                if isinstance(event, BlockStored):
                    for block_hash in event.block_hashes:
                        if block_hash not in block_locations:
                            block_locations[block_hash] = set()
                        block_locations[block_hash].add(name)

                elif isinstance(event, BlockRemoved):
                    for block_hash in event.block_hashes:
                        if block_hash in block_locations:
                            block_locations[block_hash].discard(name)
                            if not block_locations[block_hash]:
                                del block_locations[block_hash]

            print(f"Total unique blocks: {len(block_locations)}")
</syntaxhighlight>

=== Monitoring Dashboard Data ===
<syntaxhighlight lang="python">
from collections import deque
import time

class CacheMonitor:
    def __init__(self, window_seconds=60):
        self.window = window_seconds
        self.events = deque()

    def add_event(self, event, timestamp):
        self.events.append((event, timestamp))

        # Remove old events outside window
        cutoff = timestamp - self.window
        while self.events and self.events[0][1] < cutoff:
            self.events.popleft()

    def get_metrics(self):
        stores = sum(1 for e, _ in self.events if isinstance(e, BlockStored))
        removes = sum(1 for e, _ in self.events if isinstance(e, BlockRemoved))
        clears = sum(1 for e, _ in self.events if isinstance(e, AllBlocksCleared))

        return {
            "stores_per_min": stores,
            "removes_per_min": removes,
            "clears_per_min": clears,
            "net_growth": stores - removes,
        }

monitor = CacheMonitor()

while True:
    if sub.poll(50):
        _, seq_bytes, payload = sub.recv_multipart()
        event_batch = decoder.decode(payload)

        for event in event_batch.events:
            monitor.add_event(event, event_batch.ts)

        # Print metrics every 10 seconds
        if time.time() % 10 < 0.1:
            metrics = monitor.get_metrics()
            print(f"Cache activity (last 60s): {metrics}")
</syntaxhighlight>

== Reliability Features ==

=== Gap Detection ===
The example detects missed messages via sequence numbers:
<syntaxhighlight lang="python">
if last_seq >= 0 and seq > last_seq + 1:
    missed = seq - last_seq - 1
    print(f"Missed {missed} messages (last: {last_seq}, current: {seq})")
</syntaxhighlight>

=== Replay Recovery ===
When gaps are detected, request replay:
<syntaxhighlight lang="python">
replay = context.socket(zmq.REQ)
replay.connect("tcp://localhost:5558")

# Request messages starting from last_seq + 1
replay.send((last_seq + 1).to_bytes(8, "big"))

while poller.poll(timeout=200):
    seq_bytes, replay_payload = replay.recv_multipart()
    if not replay_payload:
        break  # End of replay marker

    replay_seq = int.from_bytes(seq_bytes, "big")
    event_batch = decoder.decode(replay_payload)
    process_event(event_batch)

    if replay_seq >= seq - 1:
        break  # Caught up
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
while True:
    try:
        if sub.poll(50):
            _, seq_bytes, payload = sub.recv_multipart()
            seq = int.from_bytes(seq_bytes, "big")

            event_batch = decoder.decode(payload)
            process_event(event_batch)

    except KeyboardInterrupt:
        print("Interrupted")
        break

    except Exception as e:
        print(f"Error decoding message: {e}")
        # Continue processing, don't crash on bad messages
</syntaxhighlight>

== Server Configuration ==

=== Enabling KV Event Publishing ===
<syntaxhighlight lang="bash">
# Enable KV cache event publishing
vllm serve model_name \
    --kv-cache-events-pub-addr tcp://*:5557 \
    --kv-cache-events-replay-addr tcp://*:5558

# Custom ports
vllm serve model_name \
    --kv-cache-events-pub-addr tcp://*:6000 \
    --kv-cache-events-replay-addr tcp://*:6001

# Bind to specific interface
vllm serve model_name \
    --kv-cache-events-pub-addr tcp://192.168.1.10:5557 \
    --kv-cache-events-replay-addr tcp://192.168.1.10:5558
</syntaxhighlight>

=== Security Considerations ===
* ZMQ sockets have no authentication by default
* Use firewall rules to restrict access
* Consider ZMQ's CURVE security for encryption
* Bind to localhost for local-only access

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Concept:vllm-project_vllm_KV_Cache]]
* [[related::Concept:vllm-project_vllm_Prefix_Caching]]
* [[related::Concept:vllm-project_vllm_V1_Engine]]
* [[related::Tool:ZeroMQ]]
