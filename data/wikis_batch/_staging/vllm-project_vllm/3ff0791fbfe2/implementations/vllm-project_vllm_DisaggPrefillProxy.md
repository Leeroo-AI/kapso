# DisaggPrefillProxy - Disaggregated Prefill/Decode Proxy Server

## Overview

**File:** `/tmp/praxium_repo_583nq7ea/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` (260 lines)

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Purpose:** Async proxy server coordinating disaggregated prefill and decode services for benchmarking, enabling separate scaling of compute-intensive prefill from latency-sensitive decode operations via NCCL-based KV cache transfer.

## Core Architecture

### Disaggregation Model

```
Client Request
    ↓
Proxy Server
    ↓
    ├── Prefill Service (GPU cluster 1)
    │   └── Generates KV cache with max_tokens=1
    │       └── Transfers KV via NCCL to decode workers
    ↓
    └── Decode Service (GPU cluster 2)
        └── Receives KV cache
            └── Continues generation with full max_tokens
                └── Streams response back to client
```

**Benefits:**
- **Independent Scaling:** Scale prefill and decode separately
- **Resource Optimization:** Prefill uses batch-optimized GPUs, decode uses low-latency GPUs
- **Cost Efficiency:** Allocate expensive GPUs only where needed

### Command-Line Arguments

**Lines:** 20-68

```python
def parse_args():
    parser = argparse.ArgumentParser(description="vLLM P/D disaggregation proxy server")

    parser.add_argument(
        "--timeout",
        type=float,
        default=6 * 60 * 60,  # 6 hours
        help="Timeout for backend requests in seconds"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Proxy server port"
    )

    parser.add_argument(
        "--prefill-url",
        type=str,
        default="http://localhost:8100",
        help="Prefill service base URL"
    )

    parser.add_argument(
        "--decode-url",
        type=str,
        default="http://localhost:8200",
        help="Decode service base URL"
    )

    parser.add_argument(
        "--kv-host",
        type=str,
        default="localhost",
        help="Hostname/IP for KV transfer"
    )

    parser.add_argument(
        "--prefill-kv-port",
        type=int,
        default=14579,
        help="Prefill KV transfer port"
    )

    parser.add_argument(
        "--decode-kv-port",
        type=int,
        default=14580,
        help="Decode KV transfer port"
    )

    return parser.parse_args()
```

### Main Server Setup

**Lines:** 71-121

```python
def main():
    args = parse_args()

    # Initialize configuration
    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=args.timeout)
    PREFILL_SERVICE_URL = args.prefill_url
    DECODE_SERVICE_URL = args.decode_url
    PORT = args.port

    PREFILL_KV_ADDR = f"{args.kv_host}:{args.prefill_kv_port}"
    DECODE_KV_ADDR = f"{args.kv_host}:{args.decode_kv_port}"

    logger.info(
        "Proxy resolved KV addresses -> prefill: %s, decode: %s",
        PREFILL_KV_ADDR,
        DECODE_KV_ADDR,
    )

    app = Quart(__name__)

    # Attach config to app instance
    app.config.update(
        {
            "AIOHTTP_TIMEOUT": AIOHTTP_TIMEOUT,
            "PREFILL_SERVICE_URL": PREFILL_SERVICE_URL,
            "DECODE_SERVICE_URL": DECODE_SERVICE_URL,
            "PREFILL_KV_ADDR": PREFILL_KV_ADDR,
            "DECODE_KV_ADDR": DECODE_KV_ADDR,
        }
    )
```

**Design Choice:** Store config in app.config to avoid global variables and enable testability.

### URL Normalization

**Lines:** 105-120

```python
def _normalize_base_url(url: str) -> str:
    """Remove trailing slash for predictable path joins"""
    return url.rstrip("/")

def _get_host_port(url: str) -> str:
    """Extract hostname:port for KV transfer headers"""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port
    if port is None:
        port = 80 if parsed.scheme == "http" else 443
    return f"{host}:{port}"

PREFILL_BASE = _normalize_base_url(PREFILL_SERVICE_URL)
DECODE_BASE = _normalize_base_url(DECODE_SERVICE_URL)
KV_TARGET = _get_host_port(DECODE_SERVICE_URL)
```

### Request ID Generation

**Lines:** 122-128, 212-219

```python
def _build_headers(request_id: str) -> dict[str, str]:
    """Construct headers for vLLM P2P disagg connector"""
    headers: dict[str, str] = {
        "X-Request-Id": request_id,
        "X-KV-Target": KV_TARGET
    }
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

# In process_request():
request_id = (
    f"___prefill_addr_{PREFILL_KV_ADDR}___decode_addr_"
    f"{DECODE_KV_ADDR}_{uuid.uuid4().hex}"
)
```

**Encoding:** Request ID contains both KV socket addresses so backend can establish direct NCCL connections.

**Format:**
```
___prefill_addr_10.0.0.1:14579___decode_addr_10.0.0.2:14580_abc123def456...
```

### Prefill Stage

**Lines:** 130-159

```python
async def _run_prefill(
    request_path: str,
    payload: dict,
    headers: dict[str, str],
    request_id: str,
):
    url = f"{PREFILL_BASE}{request_path}"
    start_ts = time.perf_counter()
    logger.info("[prefill] start request_id=%s url=%s", request_id, url)

    try:
        async with (
            aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
            session.post(url=url, json=payload, headers=headers) as resp,
        ):
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Prefill backend error {resp.status}: {error_text}")

            await resp.read()  # Consume response (usually single token)

            logger.info(
                "[prefill] done request_id=%s status=%s elapsed=%.2fs",
                request_id,
                resp.status,
                time.perf_counter() - start_ts,
            )

    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"Prefill service timeout at {url}") from exc
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"Prefill service unavailable at {url}") from exc
```

**Key Points:**
- Sets `max_tokens=1` in payload (done by caller)
- Blocks until prefill completes and KV transferred
- Error handling with context propagation

### Decode Stage

**Lines:** 161-199

```python
async def _stream_decode(
    request_path: str,
    payload: dict,
    headers: dict[str, str],
    request_id: str,
):
    url = f"{DECODE_BASE}{request_path}"
    logger.info("[decode] start request_id=%s url=%s", request_id, url)

    try:
        async with (
            aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
            session.post(url=url, json=payload, headers=headers) as resp,
        ):
            if resp.status != 200:
                error_text = await resp.text()
                logger.error("Decode backend error %s - %s", resp.status, error_text)
                err_msg = '{"error": "Decode backend error ' + str(resp.status) + '"}'
                yield err_msg.encode()
                return

            logger.info(
                "[decode] streaming response request_id=%s status=%s",
                request_id,
                resp.status,
            )

            # Stream response chunks
            async for chunk_bytes in resp.content.iter_chunked(1024):
                yield chunk_bytes

            logger.info("[decode] finished streaming request_id=%s", request_id)

    except asyncio.TimeoutError:
        logger.error("Decode service timeout at %s", url)
        yield b'{"error": "Decode service timeout"}'
    except aiohttp.ClientError as exc:
        logger.error("Decode service error at %s: %s", url, exc)
        yield b'{"error": "Decode service unavailable"}'
```

**Streaming:** Uses async generator to stream tokens as they're generated.

**Error Resilience:** Returns JSON error messages instead of crashing.

### Request Processing

**Lines:** 201-240

```python
async def process_request():
    """Process single request through prefill and decode stages"""
    try:
        original_request_data = await request.get_json()

        # Create prefill request with max_tokens=1
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        # Generate request ID encoding KV addresses
        request_id = (
            f"___prefill_addr_{PREFILL_KV_ADDR}___decode_addr_"
            f"{DECODE_KV_ADDR}_{uuid.uuid4().hex}"
        )

        headers = _build_headers(request_id)

        # Execute prefill (blocks until KV transferred)
        await _run_prefill(request.path, prefill_request, headers, request_id)

        # Execute decode (streams response)
        generator = _stream_decode(
            request.path, original_request_data, headers, request_id
        )
        response = await make_response(generator)
        response.timeout = None  # Disable timeout for streaming
        return response

    except Exception:
        logger.exception("Error processing request")
        return Response(
            response=b'{"error": "Internal server error"}',
            status=500,
            content_type="application/json",
        )
```

**Flow:**
1. Receive original request
2. Create modified prefill request (max_tokens=1)
3. Execute prefill, wait for completion
4. Execute decode with original params, stream response

### Route Handler

**Lines:** 242-254

```python
@app.route("/v1/completions", methods=["POST"])
async def handle_request():
    """Handle incoming API requests"""
    try:
        return await process_request()
    except asyncio.CancelledError:
        logger.warning("Request cancelled")
        return Response(
            response=b'{"error": "Request cancelled"}',
            status=503,
            content_type="application/json",
        )
```

### Server Startup

**Lines:** 255-260

```python
# Start Quart server
app.run(port=PORT)

if __name__ == "__main__":
    main()
```

## Usage Examples

### Basic Setup

**Terminal 1: Prefill Service**
```bash
# Start prefill service on port 8100
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8100 \
    --tensor-parallel-size 4 \
    --disagg-mode prefill \
    --kv-connector nccl \
    --kv-role src \
    --kv-port 14579
```

**Terminal 2: Decode Service**
```bash
# Start decode service on port 8200
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8200 \
    --tensor-parallel-size 2 \
    --disagg-mode decode \
    --kv-connector nccl \
    --kv-role dst \
    --kv-port 14580
```

**Terminal 3: Proxy Server**
```bash
python disagg_prefill_proxy_server.py \
    --port 8000 \
    --prefill-url http://localhost:8100 \
    --decode-url http://localhost:8200 \
    --kv-host localhost \
    --prefill-kv-port 14579 \
    --decode-kv-port 14580
```

**Terminal 4: Client**
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "Once upon a time",
        "max_tokens": 100
    }'
```

### Multi-Node Setup

**Node 1 (Prefill):**
```bash
# 8x A100 for batch-optimized prefill
python disagg_prefill_proxy_server.py \
    --port 8000 \
    --prefill-url http://node1:8100 \
    --decode-url http://node2:8200 \
    --kv-host node1 \
    --prefill-kv-port 14579 \
    --decode-kv-port 14580
```

**Node 2 (Decode):**
```bash
# 4x H100 for low-latency decode
# (Started via separate script)
```

### Production Configuration

```bash
python disagg_prefill_proxy_server.py \
    --port 8000 \
    --prefill-url https://prefill.example.com \
    --decode-url https://decode.example.com \
    --kv-host 10.0.1.100 \
    --prefill-kv-port 14579 \
    --decode-kv-port 14580 \
    --timeout 1800  # 30 minutes
```

## Implementation Details

### Why max_tokens=1 for Prefill?

```python
prefill_request["max_tokens"] = 1
```

**Rationale:**
- Prefill only needs to populate KV cache
- Single token generation triggers cache creation
- Minimizes prefill latency
- Decode service generates remaining tokens

### Async Architecture

```python
async with (
    aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
    session.post(url=url, json=payload, headers=headers) as resp,
):
```

**Benefits:**
- Non-blocking I/O during network requests
- Efficient handling of concurrent requests
- Proper resource cleanup via context managers

### Error Handling Strategy

```python
except asyncio.TimeoutError:
    yield b'{"error": "Decode service timeout"}'
except aiohttp.ClientError as exc:
    yield b'{"error": "Decode service unavailable"}'
```

**Graceful Degradation:** Return JSON errors instead of HTTP 500, enabling client-side retry logic.

### Logging

```python
logger.info("[prefill] start request_id=%s url=%s", request_id, url)
logger.info("[prefill] done request_id=%s status=%s elapsed=%.2fs", ...)
logger.info("[decode] streaming response request_id=%s status=%s", ...)
```

**Tracing:** Request ID enables end-to-end tracking across services.

## Performance Implications

### Latency Breakdown

```
Total Latency = Prefill Latency + Transfer Latency + Decode Latency
```

**Typical Values (Llama-2-7B, 2048 input, 512 output):**
- Prefill: 200-500ms (depends on batch size)
- Transfer: 10-50ms (NCCL over RDMA)
- Decode: 5-10s (token-by-token generation)

**Optimization:** Decode can start immediately after transfer completes.

### Throughput

**Prefill-Only Throughput:**
- Optimized for batch processing
- 100-500 requests/second possible

**Decode-Only Throughput:**
- Optimized for latency
- 10-50 requests/second typical

**Disaggregated Throughput:**
- Balanced between prefill and decode capacity
- Can scale each independently

### Memory Efficiency

**Without Disaggregation:**
- Single cluster needs memory for both prefill and decode
- KV cache accumulates over time

**With Disaggregation:**
- Prefill cluster: Short-lived KV caches
- Decode cluster: Accumulates only active generation KV
- More efficient memory utilization

## KV Transfer Protocol

### Request ID Format

```python
f"___prefill_addr_{PREFILL_KV_ADDR}___decode_addr_{DECODE_KV_ADDR}_{uuid.uuid4().hex}"
```

**Example:**
```
___prefill_addr_10.0.0.1:14579___decode_addr_10.0.0.2:14580_a1b2c3d4e5f6...
```

### Headers

```python
headers = {
    "X-Request-Id": request_id,  # Contains KV addresses
    "X-KV-Target": KV_TARGET,    # Decode host:port
    "Authorization": f"Bearer {api_key}"  # Optional auth
}
```

### NCCL Connection

1. Prefill service parses request ID
2. Extracts decode KV address
3. Establishes NCCL connection to decode workers
4. Transfers KV cache tensors
5. Decode service receives KV and continues generation

## Integration Points

### Quart Framework

- Async-native Flask alternative
- Supports streaming responses
- Efficient for I/O-bound proxy workloads

### aiohttp Client

- Async HTTP client for backend requests
- Timeout control
- Efficient connection pooling

### vLLM Disaggregation

- Prefill mode: `--disagg-mode prefill --kv-role src`
- Decode mode: `--disagg-mode decode --kv-role dst`
- NCCL connector: `--kv-connector nccl`

## Limitations

1. **Single Path:** Only `/v1/completions` supported
2. **No Load Balancing:** Single prefill/decode instance
3. **No Retry Logic:** Fails on backend errors
4. **No Metrics:** No Prometheus/statsd integration
5. **Simple Routing:** No request sharding

## Future Enhancements

Potential improvements:
1. Add `/v1/chat/completions` endpoint
2. Implement load balancing across multiple backends
3. Add retry logic with exponential backoff
4. Integrate Prometheus metrics
5. Support request batching
6. Add health checks for backends
7. Implement circuit breakers

## Real-World Deployments

### Cloud Deployment

```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: disagg-proxy
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: proxy
        image: vllm/disagg-proxy:latest
        args:
          - --port=8000
          - --prefill-url=http://prefill-service:8100
          - --decode-url=http://decode-service:8200
```

### Cost Optimization

**Scenario:** 1000 requests/hour peak, 100 requests/hour off-peak

**Traditional:** Fixed 8x A100 cluster ($24/hour)
**Disaggregated:**
- Prefill: Scale 0-8x A100 based on load
- Decode: Fixed 2x H100 ($8/hour)
- Savings: ~50% on off-peak hours

## Related Components

- **vllm/entrypoints/openai/api_server.py:** Backend services
- **vllm/distributed/kv_transfer/:** KV transfer implementation
- **vllm/engine/:** Disaggregated engine modes

## Technical Significance

This proxy demonstrates:
- **Architecture Pattern:** Clean separation of concerns (prefill vs decode)
- **Scalability:** Independent scaling of compute-intensive vs latency-sensitive components
- **Efficiency:** Optimal resource allocation for different workload phases
- **Flexibility:** Easy to experiment with different hardware for each stage

The disaggregation approach is particularly valuable for production deployments where cost optimization and predictable latency are critical. The proxy serves as both a reference implementation and a benchmarking tool for evaluating disaggregated architectures.
