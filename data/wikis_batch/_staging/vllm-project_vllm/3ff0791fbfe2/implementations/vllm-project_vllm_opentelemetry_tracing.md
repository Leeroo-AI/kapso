# Implementation: OpenTelemetry Tracing Integration

**File:** `/tmp/praxium_repo_583nq7ea/vllm/tracing.py` (135 lines)
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The `tracing.py` module provides OpenTelemetry (OTEL) integration for distributed tracing in vLLM. It enables observability through span-based tracing with graceful degradation when OTEL packages are unavailable. The module handles trace context propagation, exporter configuration, and defines LLM-specific semantic conventions for span attributes.

**Key Components:**
- `init_tracer()`: Initializes OTEL tracer with OTLP exporter
- `extract_trace_context()`: Extracts trace context from HTTP headers
- `SpanAttributes`: LLM-specific semantic conventions
- Graceful degradation when OTEL is not installed
- Support for both gRPC and HTTP/protobuf OTLP protocols

## Implementation Details

### Import Strategy with Graceful Degradation

```python
_is_otel_imported = False
otel_import_error_traceback: str | None = None
try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    # See https://github.com/vllm-project/vllm/pull/7266#discussion_r1707395458
    import traceback

    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass
```

**Design Pattern:**

1. **Try-Import with Flag**: Sets `_is_otel_imported = True` on success
2. **Capture Traceback**: Stores import error details as string
3. **Stub Classes**: Provides type stubs when OTEL unavailable
4. **Memory Safety**: Only stores string, not exception object (prevents reference cycles)

**Why String Traceback?**
```python
# Storing exception directly can cause memory leaks
# Exception objects hold references to local variables and frames
otel_import_error_traceback = traceback.format_exc()  # String only
```

### Availability Check

```python
def is_otel_available() -> bool:
    return _is_otel_imported
```

**Usage:**
```python
if is_otel_available():
    tracer = init_tracer("vllm", endpoint)
else:
    logger.warning("OTEL not available, tracing disabled")
```

### Tracer Initialization

```python
def init_tracer(
    instrumenting_module_name: str, otlp_traces_endpoint: str
) -> Tracer | None:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )
    trace_provider = TracerProvider()

    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer
```

**Initialization Flow:**

1. **Check Availability**: Raises helpful error if OTEL unavailable
2. **Create Provider**: TracerProvider manages tracer lifecycle
3. **Configure Exporter**: Get appropriate exporter (gRPC/HTTP)
4. **Add Processor**: BatchSpanProcessor for efficient batching
5. **Set Global Provider**: Makes tracer available globally
6. **Get Tracer**: Named tracer for this module

**Parameters:**
- `instrumenting_module_name`: Identifies the source (e.g., "vllm.engine")
- `otlp_traces_endpoint`: OTLP collector endpoint (e.g., "http://localhost:4317")

### Span Exporter Configuration

```python
def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter(endpoint=endpoint)
```

**Protocol Selection:**

1. **Environment Variable**: `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`
   - `"grpc"` (default): gRPC protocol
   - `"http/protobuf"`: HTTP with protobuf encoding

2. **Lazy Import**: Only imports needed exporter
3. **Validation**: Raises error for unsupported protocols

**Supported Protocols:**

**gRPC Protocol:**
- Default choice
- Efficient binary protocol
- Bidirectional streaming
- Better for high-volume tracing

**HTTP/Protobuf Protocol:**
- Firewall-friendly (uses standard HTTP)
- Easier to debug (can inspect with curl)
- Works with more proxy configurations

### Trace Context Extraction

```python
TRACE_HEADERS = ["traceparent", "tracestate"]
```

**W3C Trace Context Headers:**
- `traceparent`: Trace ID, span ID, flags (required)
- `tracestate`: Vendor-specific state (optional)

```python
def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None
```

**Functionality:**
- Extracts W3C Trace Context from HTTP headers
- Returns OTEL Context object (or None)
- Returns None if OTEL unavailable

**Usage:**
```python
# In HTTP server
incoming_headers = request.headers
trace_context = extract_trace_context(incoming_headers)

# Start span with parent context
with tracer.start_as_current_span("handle_request", context=trace_context):
    # Request handling
    pass
```

**Trace Context Format (traceparent):**
```
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
             └─ version
                └─ trace-id (16 bytes hex)
                                                └─ span-id (8 bytes hex)
                                                                    └─ flags
```

### Header Utilities

```python
def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}
```

**Purpose:** Extract only tracing-related headers for propagation

**Usage:**
```python
# Propagate trace to downstream service
trace_headers = extract_trace_headers(incoming_headers)
response = requests.get(downstream_url, headers=trace_headers)
```

```python
def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)
```

**Purpose:** Quick check if request has trace context

**Usage:**
```python
if contains_trace_headers(request.headers):
    if not tracing_enabled:
        log_tracing_disabled_warning()
```

### Semantic Conventions: SpanAttributes

```python
class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
```

**Standard Semantic Conventions:**
- Based on OTEL GenAI semantic conventions
- Enables consistent metrics across LLM systems
- Compatible with standard OTEL tooling

```python
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"
```

**vLLM-Specific Extensions:**
- Additional latency breakdowns
- Request-specific identifiers
- Multi-sequence support
- Detailed pipeline timing

### Warning Utility

```python
@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")
```

**Design:**
- `@run_once` decorator: Only logs first occurrence
- Prevents log spam when tracing is disabled
- Informs about potential configuration issue

## Usage Patterns

### Basic Initialization

```python
from vllm.tracing import init_tracer, is_otel_available

if is_otel_available():
    tracer = init_tracer(
        instrumenting_module_name="vllm.engine",
        otlp_traces_endpoint="http://localhost:4317"
    )
else:
    logger.warning("Tracing disabled: OpenTelemetry not installed")
    tracer = None
```

### Creating Spans

```python
if tracer:
    with tracer.start_as_current_span("generate_text") as span:
        # Set attributes
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, 100)
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE, 0.8)

        # Generate text
        output = generate(...)

        # Set result attributes
        span.set_attribute(
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
            len(output.token_ids)
        )
```

### Trace Context Propagation

```python
from vllm.tracing import extract_trace_context

# HTTP server endpoint
@app.post("/v1/completions")
async def completions(request: Request):
    # Extract parent trace context
    trace_context = extract_trace_context(dict(request.headers))

    # Start span with parent context
    if tracer:
        with tracer.start_as_current_span(
            "handle_completion",
            context=trace_context
        ) as span:
            result = await process_completion(request)
            return result
    else:
        return await process_completion(request)
```

### Comprehensive Request Tracing

```python
def trace_request(tracer, request, output, metrics):
    if not tracer:
        return

    with tracer.start_as_current_span("llm_request") as span:
        # Request parameters
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request.request_id)
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, request.max_tokens)
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE, request.temperature)
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TOP_P, request.top_p)
        span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL, request.model)

        # Usage statistics
        span.set_attribute(
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS,
            len(request.prompt_token_ids)
        )
        span.set_attribute(
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
            len(output.token_ids)
        )

        # Latency breakdowns
        span.set_attribute(
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE,
            metrics.time_in_queue
        )
        span.set_attribute(
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN,
            metrics.first_token_time - metrics.arrival_time
        )
        span.set_attribute(
            SpanAttributes.GEN_AI_LATENCY_E2E,
            metrics.finished_time - metrics.arrival_time
        )
        span.set_attribute(
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_SCHEDULER,
            metrics.scheduler_time
        )
        span.set_attribute(
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD,
            metrics.model_forward_time
        )
        span.set_attribute(
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE,
            metrics.model_execute_time
        )
```

### Environment Configuration

```bash
# Set OTLP endpoint
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://jaeger:4317

# Set protocol (default: grpc)
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc

# Or use HTTP
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://jaeger:4318/v1/traces
```

### Integration with Request Handler

```python
class RequestHandler:
    def __init__(self, tracer=None):
        self.tracer = tracer

    async def handle_request(self, request, headers):
        # Check for trace context
        if contains_trace_headers(headers):
            if not self.tracer:
                log_tracing_disabled_warning()
                trace_context = None
            else:
                trace_context = extract_trace_context(headers)
        else:
            trace_context = None

        # Process with tracing
        if self.tracer:
            with self.tracer.start_as_current_span(
                "process_request",
                context=trace_context
            ) as span:
                return await self._process(request, span)
        else:
            return await self._process(request, None)

    async def _process(self, request, span):
        # Processing logic
        if span:
            span.set_attribute("request_id", request.id)
        # ...
```

## Integration Points

### LLM Engine

```python
class LLMEngine:
    def __init__(self, ...):
        if tracing_enabled:
            self.tracer = init_tracer("vllm.engine", otlp_endpoint)
        else:
            self.tracer = None

    def generate(self, request):
        if self.tracer:
            with self.tracer.start_as_current_span("engine.generate") as span:
                span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request.id)
                return self._generate(request, span)
        else:
            return self._generate(request, None)
```

### OpenAI API Server

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    trace_context = extract_trace_context(dict(request.headers))

    if tracer:
        with tracer.start_as_current_span(
            "chat_completion",
            context=trace_context
        ) as span:
            result = await process_chat_completion(body)
            # Set span attributes from result
            return result
```

### Distributed Inference

```python
# Worker A (coordinator)
with tracer.start_as_current_span("distributed_inference") as parent_span:
    trace_headers = extract_trace_headers({"traceparent": parent_span.context})

    # Send to worker B with trace context
    response = requests.post(
        "http://worker-b/forward",
        headers=trace_headers,
        json=inputs
    )

# Worker B (receives trace context)
@app.post("/forward")
async def forward(request: Request):
    trace_context = extract_trace_context(dict(request.headers))

    with tracer.start_as_current_span("worker_forward", context=trace_context):
        # This span is child of parent_span from Worker A
        result = model.forward(...)
        return result
```

## Observability Backends

### Jaeger

```bash
# Start Jaeger all-in-one
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Configure vLLM
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc

# View traces at http://localhost:16686
```

### Zipkin

```bash
# Start Zipkin
docker run -d --name zipkin \
  -p 9411:9411 \
  openzipkin/zipkin

# Configure vLLM for HTTP
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:9411/api/v2/spans
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
```

### Grafana Tempo

```yaml
# tempo-config.yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317

# Configure vLLM
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://tempo:4317
```

### Cloud Providers

**Google Cloud Trace:**
```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://cloudtrace.googleapis.com
# Also need authentication
```

**AWS X-Ray:**
```bash
# Use OTEL Collector as intermediary
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://otel-collector:4317
# Collector configured to export to X-Ray
```

## Performance Considerations

### Batch Span Processor

```python
trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
```

**Batching Strategy:**
- Accumulates spans in memory
- Exports in batches (default: 512 spans or 5 seconds)
- Reduces network overhead
- Prevents blocking request processing

**Configuration:**
```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor

processor = BatchSpanProcessor(
    span_exporter,
    max_queue_size=2048,        # Default: 2048
    schedule_delay_millis=5000, # Default: 5000ms
    max_export_batch_size=512,  # Default: 512
    export_timeout_millis=30000 # Default: 30000ms
)
```

### Overhead

**Span Creation:**
- ~1-10 microseconds per span
- Negligible for LLM requests (milliseconds to seconds)

**Attribute Setting:**
- ~100 nanoseconds per attribute
- Even hundreds of attributes have minimal impact

**Export:**
- Async/background process
- Doesn't block request handling
- Network I/O happens separately

### Sampling

For very high-throughput systems, consider sampling:
```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Trace 10% of requests
sampler = TraceIdRatioBased(0.1)

trace_provider = TracerProvider(sampler=sampler)
```

## Design Rationale

### Why Graceful Degradation?

**Alternative:** Make OTEL a required dependency
**Problem:** Increases install size, not all users need tracing

**Chosen:** Optional dependency with graceful degradation
**Benefits:**
- vLLM works without OTEL installed
- Users opt-in to tracing
- Smaller default installation

### Why String Traceback?

```python
otel_import_error_traceback = traceback.format_exc()  # String
```

**Alternative:** Store exception object
**Problem:** Memory leaks from reference cycles

**Chosen:** Store string representation
**Benefits:**
- No reference cycles
- Still provides debugging information
- Minimal memory footprint

### Why BatchSpanProcessor?

**Alternative:** SimpleSpanProcessor (exports immediately)
**Problem:** Blocks on network I/O for each span

**Chosen:** BatchSpanProcessor
**Benefits:**
- Async export
- Reduced network overhead
- Better performance under load

### Why W3C Trace Context?

**Alternative:** Proprietary trace propagation
**Problem:** Incompatible with standard tools

**Chosen:** W3C Trace Context standard
**Benefits:**
- Interoperable with all OTEL-compatible systems
- Industry standard
- Supported by all major observability platforms

## Error Handling

### OTEL Not Installed

```python
if not is_otel_available():
    logger.warning("OpenTelemetry not installed, tracing disabled")
    tracer = None

# Code handles tracer=None gracefully
if tracer:
    with tracer.start_as_current_span(...):
        # traced
        pass
else:
    # not traced
    pass
```

### Invalid Endpoint

```python
try:
    tracer = init_tracer("vllm", "invalid-endpoint")
except Exception as e:
    logger.error(f"Failed to initialize tracer: {e}")
    tracer = None
```

### Export Failures

BatchSpanProcessor handles export failures:
- Retries with backoff
- Drops spans if queue full
- Logs errors (doesn't crash application)

## Testing Considerations

### Mock Tracer

```python
from unittest.mock import Mock

def test_with_tracing():
    mock_tracer = Mock()
    mock_span = Mock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

    # Test code with mock tracer
    result = generate_with_tracing(mock_tracer, request)

    # Verify span created
    mock_tracer.start_as_current_span.assert_called_once()
    mock_span.set_attribute.assert_called()
```

### Test Without OTEL

```python
def test_without_otel():
    # Should work even if OTEL not installed
    if not is_otel_available():
        tracer = None
    else:
        tracer = init_tracer("test", "http://localhost:4317")

    result = generate(request, tracer)
    assert result is not None
```

### Integration Test

```python
def test_trace_propagation():
    # Start test OTEL collector
    collector = start_test_collector()

    # Initialize tracer
    tracer = init_tracer("test", collector.endpoint)

    # Make traced request
    with tracer.start_as_current_span("test_span") as span:
        span.set_attribute("test_key", "test_value")

    # Wait for export
    time.sleep(1)

    # Verify trace received
    traces = collector.get_traces()
    assert len(traces) == 1
    assert traces[0].attributes["test_key"] == "test_value"
```

## Related Components

- **vllm.engine.llm_engine**: Uses tracer for request tracing
- **vllm.entrypoints.openai**: Propagates trace context from HTTP
- **vllm.sequence.RequestMetrics**: Provides timing data for spans
- **vllm.logger**: Logging integration with tracing

## Future Enhancements

1. **Automatic Instrumentation**: Auto-trace all requests without code changes
2. **Metrics Integration**: Combine traces with Prometheus metrics
3. **Logs Correlation**: Link logs to traces via trace IDs
4. **Custom Samplers**: Intelligent sampling based on request properties
5. **Trace Analysis**: Built-in tools for trace analysis and visualization

## Summary

The `tracing.py` module provides production-ready distributed tracing for vLLM through OpenTelemetry integration. Its graceful degradation design ensures vLLM works without OTEL while enabling powerful observability when needed. The comprehensive span attributes and support for W3C Trace Context make it compatible with industry-standard observability platforms, enabling deep insights into LLM serving performance and behavior.
