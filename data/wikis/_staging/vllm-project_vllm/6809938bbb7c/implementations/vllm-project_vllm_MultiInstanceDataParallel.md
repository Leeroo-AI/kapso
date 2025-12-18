{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Online Serving]], [[domain::Data Parallelism]], [[domain::Multi-Instance]], [[domain::Distributed Serving]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates multi-instance data-parallel serving where independent vLLM instances coordinate to distribute requests across multiple model replicas for higher throughput.

=== Description ===
This example shows how to deploy multiple vLLM server instances in data-parallel mode, where each instance runs independently but coordinates request routing through a shared RPC layer. Unlike single-process data parallelism, this approach allows running separate server processes (potentially on different machines) that appear as a single logical service.

The pattern uses:
* Separate vLLM processes for isolation and fault tolerance
* RPC-based coordination for request routing
* Explicit rank selection for directing requests to specific instances
* Background logging for monitoring aggregate throughput
* AsyncLLMEngine for asynchronous request handling

This enables building high-throughput serving systems where requests can be distributed across multiple model replicas, with each replica handling different subsets of traffic.

=== Usage ===
Use this approach when:
* Building high-availability serving systems with multiple replicas
* Scaling beyond single-process throughput limits
* Deploying across multiple machines or nodes
* Implementing custom load balancing strategies
* Requiring process isolation for stability or resource management
* Separating control plane (request router) from data plane (model servers)
* Supporting dynamic scaling of model instances

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/multi_instance_data_parallel.py examples/online_serving/multi_instance_data_parallel.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Terminal 1: Start headless server instance (DP rank 1)
vllm serve ibm-research/PowerMoE-3b \
    -dp 2 -dpr 1 \
    --data-parallel-address 127.0.0.1 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --enforce-eager \
    --headless

# Terminal 2: Run client that routes to rank 1
python examples/online_serving/multi_instance_data_parallel.py

# The client will wait for handshake with the headless server,
# then send requests to DP rank 1
</syntaxhighlight>

== Key Concepts ==

=== Multi-Instance Architecture ===
The pattern involves multiple components:

'''Headless Server(s):'''
* Run with <code>--headless</code> flag
* No HTTP API exposed
* Pure inference workers
* Register with coordinator via RPC

'''Client/Coordinator:'''
* Creates AsyncLLMEngine
* Connects to headless servers via RPC
* Routes requests to specific DP ranks
* Can run on separate machine from servers

'''RPC Coordination:'''
* <code>--data-parallel-address</code>: IP for RPC communication
* <code>--data-parallel-rpc-port</code>: Port for coordination
* <code>--data-parallel-size</code> (-dp): Total number of instances
* <code>--data-parallel-rank</code> (-dpr): Instance ID

=== Request Routing ===
The client explicitly routes requests to ranks:
<syntaxhighlight lang="python">
async for output in engine_client.generate(
    prompt=prompt,
    sampling_params=sampling_params,
    request_id=f"request-{i}",
    data_parallel_rank=1,  # Route to specific rank
):
    final_output = output
</syntaxhighlight>

This enables:
* Custom load balancing policies
* Affinity-based routing (sticky sessions)
* Priority-based assignment
* Geographic routing

=== Background Logging ===
The example demonstrates background metrics collection:
<syntaxhighlight lang="python">
from vllm.v1.metrics.loggers import AggregatedLoggingStatLogger

engine_client = AsyncLLMEngine.from_engine_args(
    engine_args,
    stat_loggers=[AggregatedLoggingStatLogger],
)

# Run logging in background thread
def _do_background_logging(engine, interval, stop_event):
    while not stop_event.is_set():
        asyncio.run(engine.do_log_stats())
        stop_event.wait(interval)

logging_thread = threading.Thread(
    target=_do_background_logging,
    args=(engine_client, 5, stop_logging_event),
    daemon=True,
)
logging_thread.start()
</syntaxhighlight>

=== Handshake and Coordination ===
Instances perform handshake during startup:
1. Each instance registers with coordinator
2. Coordinator waits for all instances (dp_size)
3. Handshake completes when all connected
4. Requests can begin flowing

This ensures consistent state before serving.

== Usage Examples ==

=== Basic Two-Instance Setup ===
<syntaxhighlight lang="bash">
# Terminal 1: Server instance 0
vllm serve microsoft/phi-2 \
    -dp 2 -dpr 0 \
    --data-parallel-address 127.0.0.1 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --headless

# Terminal 2: Server instance 1
vllm serve microsoft/phi-2 \
    -dp 2 -dpr 1 \
    --data-parallel-address 127.0.0.1 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --headless

# Terminal 3: Client coordinator
python multi_instance_client.py
</syntaxhighlight>

=== Client Implementation ===
<syntaxhighlight lang="python">
import asyncio
import threading
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.loggers import AggregatedLoggingStatLogger

async def main():
    # Create client engine that connects to headless servers
    engine_args = AsyncEngineArgs(
        model="microsoft/phi-2",
        data_parallel_size=2,
        tensor_parallel_size=1,
        data_parallel_address="127.0.0.1",
        data_parallel_rpc_port=62300,
        data_parallel_size_local=1,
        enforce_eager=True,
    )

    engine_client = AsyncLLMEngine.from_engine_args(
        engine_args,
        stat_loggers=[AggregatedLoggingStatLogger],
    )

    # Generate requests
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    # Route requests to different ranks
    for i in range(10):
        rank = i % 2  # Alternate between ranks
        async for output in engine_client.generate(
            prompt=f"Tell me about topic {i}",
            sampling_params=sampling_params,
            request_id=f"req-{i}",
            data_parallel_rank=rank,
        ):
            final_output = output

        if final_output:
            print(f"Rank {rank}: {final_output.outputs[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
</syntaxhighlight>

=== Round-Robin Load Balancing ===
<syntaxhighlight lang="python">
class RoundRobinRouter:
    def __init__(self, num_ranks):
        self.num_ranks = num_ranks
        self.current = 0

    def next_rank(self):
        rank = self.current
        self.current = (self.current + 1) % self.num_ranks
        return rank

router = RoundRobinRouter(num_ranks=2)

for i in range(100):
    rank = router.next_rank()
    async for output in engine_client.generate(
        prompt=prompts[i],
        sampling_params=sampling_params,
        request_id=f"req-{i}",
        data_parallel_rank=rank,
    ):
        pass
</syntaxhighlight>

=== Least-Loaded Routing ===
<syntaxhighlight lang="python">
import asyncio
from collections import defaultdict

class LeastLoadedRouter:
    def __init__(self, num_ranks):
        self.num_ranks = num_ranks
        self.pending_requests = defaultdict(int)
        self.lock = asyncio.Lock()

    async def acquire_rank(self):
        async with self.lock:
            # Find rank with fewest pending requests
            rank = min(range(self.num_ranks),
                      key=lambda r: self.pending_requests[r])
            self.pending_requests[rank] += 1
            return rank

    async def release_rank(self, rank):
        async with self.lock:
            self.pending_requests[rank] -= 1

router = LeastLoadedRouter(num_ranks=2)

async def process_request(prompt, request_id):
    rank = await router.acquire_rank()
    try:
        async for output in engine_client.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            data_parallel_rank=rank,
        ):
            final_output = output
        return final_output
    finally:
        await router.release_rank(rank)

# Process many requests concurrently
tasks = [process_request(p, f"req-{i}") for i, p in enumerate(prompts)]
results = await asyncio.gather(*tasks)
</syntaxhighlight>

=== Multi-Node Deployment ===
<syntaxhighlight lang="bash">
# On machine 1 (192.168.1.10)
vllm serve model_name \
    -dp 4 -dpr 0 \
    --data-parallel-address 192.168.1.10 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --headless

vllm serve model_name \
    -dp 4 -dpr 1 \
    --data-parallel-address 192.168.1.10 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --headless

# On machine 2 (192.168.1.11)
vllm serve model_name \
    -dp 4 -dpr 2 \
    --data-parallel-address 192.168.1.10 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --headless

vllm serve model_name \
    -dp 4 -dpr 3 \
    --data-parallel-address 192.168.1.10 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --headless

# Client on any machine
python client.py  # Configured with same address/port
</syntaxhighlight>

=== With Tensor Parallelism ===
<syntaxhighlight lang="bash">
# Each DP rank uses TP=2 (4 GPUs per instance, 2 instances)
# Terminal 1: DP rank 0 with TP=2
vllm serve meta-llama/Llama-3.1-70B \
    -dp 2 -dpr 0 -tp 2 \
    --data-parallel-address 127.0.0.1 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --headless

# Terminal 2: DP rank 1 with TP=2
vllm serve meta-llama/Llama-3.1-70B \
    -dp 2 -dpr 1 -tp 2 \
    --data-parallel-address 127.0.0.1 \
    --data-parallel-rpc-port 62300 \
    --data-parallel-size-local 1 \
    --headless

# Client (no TP needed, just coordinates)
# Uses 8 total GPUs: 2 DP ranks × 2 TP × 2 GPUs
</syntaxhighlight>

== Monitoring and Logging ==

=== Aggregated Statistics ===
<syntaxhighlight lang="python">
from vllm.v1.metrics.loggers import AggregatedLoggingStatLogger

engine_client = AsyncLLMEngine.from_engine_args(
    engine_args,
    stat_loggers=[AggregatedLoggingStatLogger],
)

# Start background logging
stop_event = threading.Event()
logging_thread = threading.Thread(
    target=_do_background_logging,
    args=(engine_client, 5, stop_event),
    daemon=True,
)
logging_thread.start()

# ... run inference ...

# Stop logging
stop_event.set()
logging_thread.join()
</syntaxhighlight>

=== Custom Metrics Collection ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class RankMetrics:
    requests_completed: int = 0
    tokens_generated: int = 0
    total_latency: float = 0.0

class MetricsCollector:
    def __init__(self, num_ranks):
        self.metrics = {i: RankMetrics() for i in range(num_ranks)}

    async def track_request(self, rank, prompt, request_id):
        start = time.time()

        async for output in engine_client.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            data_parallel_rank=rank,
        ):
            final_output = output

        latency = time.time() - start
        self.metrics[rank].requests_completed += 1
        self.metrics[rank].tokens_generated += len(final_output.outputs[0].token_ids)
        self.metrics[rank].total_latency += latency

        return final_output

    def print_summary(self):
        print("\n=== Per-Rank Metrics ===")
        for rank, m in self.metrics.items():
            avg_latency = m.total_latency / m.requests_completed if m.requests_completed > 0 else 0
            print(f"Rank {rank}:")
            print(f"  Requests: {m.requests_completed}")
            print(f"  Tokens: {m.tokens_generated}")
            print(f"  Avg latency: {avg_latency:.2f}s")

collector = MetricsCollector(num_ranks=2)

for i, prompt in enumerate(prompts):
    rank = i % 2
    await collector.track_request(rank, prompt, f"req-{i}")

collector.print_summary()
</syntaxhighlight>

== Fault Tolerance ==

=== Detecting Failed Instances ===
<syntaxhighlight lang="python">
import asyncio

async def generate_with_retry(prompt, request_id, rank, max_retries=3):
    for attempt in range(max_retries):
        try:
            async for output in engine_client.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=f"{request_id}-attempt-{attempt}",
                data_parallel_rank=rank,
            ):
                final_output = output
            return final_output

        except Exception as e:
            print(f"Rank {rank} failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                # Try different rank
                rank = (rank + 1) % num_ranks
                await asyncio.sleep(1)
            else:
                raise

    return None
</syntaxhighlight>

=== Health Checks ===
<syntaxhighlight lang="python">
async def check_rank_health(rank):
    try:
        # Send simple test request
        async for _ in engine_client.generate(
            prompt="test",
            sampling_params=SamplingParams(max_tokens=1),
            request_id=f"health-{rank}",
            data_parallel_rank=rank,
        ):
            pass
        return True
    except Exception:
        return False

async def monitor_health():
    while True:
        for rank in range(num_ranks):
            healthy = await check_rank_health(rank)
            print(f"Rank {rank}: {'healthy' if healthy else 'unhealthy'}")
        await asyncio.sleep(30)
</syntaxhighlight>

== Performance Considerations ==

=== Throughput Scaling ===
* Each instance provides independent throughput
* Near-linear scaling with number of instances
* No cross-instance communication during inference
* Limited only by request routing overhead

=== Latency Characteristics ===
* Individual request latency same as single instance
* Routing adds <1ms overhead
* No increase in per-request latency
* Benefits from concurrent request processing

=== Resource Utilization ===
* Each instance uses separate GPU memory
* Total memory: <code>num_instances × model_size</code>
* CPU overhead: One process per instance
* Network: RPC coordination minimal

== Comparison with Alternatives ==

{| class="wikitable"
|-
! Approach !! Isolation !! Fault Tolerance !! Complexity !! Scalability
|-
| Multi-Instance DP || High (separate processes) || Good || Medium || Excellent
|-
| Single-Process DP || Low (shared process) || Poor || Low || Good
|-
| Multiple Services || High (separate services) || Excellent || High || Excellent
|-
| Single Service || None || Poor || Low || Limited
|}

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Implementation:vllm-project_vllm_TorchrunDataParallelInference]]
* [[related::Concept:vllm-project_vllm_Data_Parallelism]]
* [[related::Concept:vllm-project_vllm_Distributed_Serving]]
* [[related::API:vllm-project_vllm_AsyncLLMEngine]]
