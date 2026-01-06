{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Observability]], [[domain::Distributed Tracing]], [[domain::Monitoring]], [[domain::OpenTelemetry]], [[domain::Performance]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
The tracing module provides OpenTelemetry integration for distributed tracing and performance monitoring of vLLM inference requests.

=== Description ===
This module enables distributed tracing in vLLM using OpenTelemetry standards. Key features include:

* '''Tracer initialization:''' Support for gRPC and HTTP/protobuf OTLP exporters
* '''Context propagation:''' Extract trace context from HTTP headers (traceparent, tracestate)
* '''Semantic conventions:''' Pre-defined span attributes for gen AI metrics
* '''Graceful degradation:''' Functions safely when OpenTelemetry is not installed
* '''Performance metrics:''' Track tokens, latency, throughput, queue time, and more

The module defines semantic conventions for generative AI spans including completion tokens, prompt tokens, temperature, top_p, time-to-first-token, end-to-end latency, and scheduler/model execution times. It handles both request-level and system-level tracing.

=== Usage ===
Use this module when you need to:
* Monitor vLLM inference performance in production
* Integrate with existing OpenTelemetry infrastructure
* Debug latency issues across distributed systems
* Track request flow through multiple services
* Analyze queue time, scheduling time, and model execution time
* Export traces to Jaeger, Zipkin, or other OTLP-compatible backends

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/tracing.py vllm/tracing.py]

=== Signature ===
<syntaxhighlight lang="python">
# Check if OpenTelemetry is available
def is_otel_available() -> bool

# Initialize tracer
def init_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str
) -> Tracer | None

# Get span exporter based on protocol
def get_span_exporter(endpoint: str) -> OTLPSpanExporter

# Extract trace context from headers
def extract_trace_context(
    headers: Mapping[str, str] | None
) -> Context | None

# Extract only trace headers
def extract_trace_headers(
    headers: Mapping[str, str]
) -> Mapping[str, str]

# Check if headers contain trace context
def contains_trace_headers(
    headers: Mapping[str, str]
) -> bool

# Log warning about disabled tracing (run once)
def log_tracing_disabled_warning() -> None

# Span attributes for gen AI
class SpanAttributes:
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"

# Constants
TRACE_HEADERS = ["traceparent", "tracestate"]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.tracing import (
    is_otel_available,
    init_tracer,
    extract_trace_context,
    extract_trace_headers,
    contains_trace_headers,
    SpanAttributes,
    TRACE_HEADERS,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| instrumenting_module_name || str || Name of module to instrument (e.g., "vllm.api_server")
|-
| otlp_traces_endpoint || str || OTLP endpoint URL for trace export
|-
| headers || dict[str, str] || HTTP headers containing trace context
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Tracer || Tracer &#124; None || OpenTelemetry tracer instance
|-
| Context || Context &#124; None || Extracted trace context for propagation
|-
| trace_headers || dict[str, str] || Filtered headers containing only trace data
|}

== Usage Examples ==

=== Initialize Tracing ===
<syntaxhighlight lang="python">
from vllm.tracing import init_tracer, is_otel_available
import os

# Check availability
if is_otel_available():
    print("OpenTelemetry is available")
else:
    print("OpenTelemetry not installed")

# Initialize tracer
tracer = init_tracer(
    instrumenting_module_name="vllm.api_server",
    otlp_traces_endpoint="http://localhost:4317"
)

# Set protocol via environment variable
os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "grpc"  # or "http/protobuf"
tracer = init_tracer("vllm.engine", "http://collector:4318/v1/traces")
</syntaxhighlight>

=== Create Spans with Attributes ===
<syntaxhighlight lang="python">
from vllm.tracing import init_tracer, SpanAttributes
from opentelemetry.trace import SpanKind

tracer = init_tracer("vllm.api", "http://localhost:4317")

# Create a span for inference request
with tracer.start_as_current_span(
    "inference_request",
    kind=SpanKind.SERVER,
) as span:
    # Set request attributes
    span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, "req-12345")
    span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL, "meta-llama/Llama-2-7b")
    span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, 100)
    span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE, 0.7)
    span.set_attribute(SpanAttributes.GEN_AI_REQUEST_TOP_P, 0.9)

    # Process request
    output = process_request(...)

    # Set output attributes
    span.set_attribute(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, 50)
    span.set_attribute(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS, 75)
</syntaxhighlight>

=== Extract and Propagate Trace Context ===
<syntaxhighlight lang="python">
from vllm.tracing import (
    extract_trace_context,
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning
)

# API server receives request with trace headers
incoming_headers = {
    "content-type": "application/json",
    "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    "tracestate": "congo=t61rcWkgMzE",
}

# Check if tracing context present
if contains_trace_headers(incoming_headers):
    if is_otel_available():
        # Extract context
        context = extract_trace_context(incoming_headers)

        # Use context to create child span
        with tracer.start_as_current_span("vllm.inference", context=context):
            # This span will be child of incoming trace
            pass
    else:
        log_tracing_disabled_warning()

# Extract only trace headers for forwarding
trace_headers = extract_trace_headers(incoming_headers)
print(trace_headers)
# {'traceparent': '00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01',
#  'tracestate': 'congo=t61rcWkgMzE'}
</syntaxhighlight>

=== Track Latency Metrics ===
<syntaxhighlight lang="python">
import time
from vllm.tracing import init_tracer, SpanAttributes

tracer = init_tracer("vllm.engine", "http://localhost:4317")

with tracer.start_as_current_span("generate_tokens") as span:
    start = time.time()

    # Track queue time
    queue_start = time.time()
    wait_in_queue()
    queue_time = time.time() - queue_start
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE, queue_time)

    # Track time to first token
    first_token_start = time.time()
    first_token = generate_first_token()
    ttft = time.time() - first_token_start
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN, ttft)

    # Track prefill time
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL, ttft)

    # Track decode time
    decode_start = time.time()
    remaining_tokens = generate_remaining_tokens()
    decode_time = time.time() - decode_start
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE, decode_time)

    # Track total E2E
    e2e = time.time() - start
    span.set_attribute(SpanAttributes.GEN_AI_LATENCY_E2E, e2e)
</syntaxhighlight>

=== Track Scheduler and Model Metrics ===
<syntaxhighlight lang="python">
from vllm.tracing import init_tracer, SpanAttributes

tracer = init_tracer("vllm.worker", "http://localhost:4317")

with tracer.start_as_current_span("model_execution") as span:
    # Track scheduler time
    scheduler_time = 0.05
    span.set_attribute(
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_SCHEDULER,
        scheduler_time
    )

    # Track model forward pass
    forward_time = 0.1
    span.set_attribute(
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD,
        forward_time
    )

    # Track total model execute (includes sync, sampling, etc.)
    execute_time = 0.15
    span.set_attribute(
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE,
        execute_time
    )

    # Track inference time
    inference_time = 0.12
    span.set_attribute(
        SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE,
        inference_time
    )
</syntaxhighlight>

=== Full Request Tracing Example ===
<syntaxhighlight lang="python">
from vllm.tracing import (
    init_tracer,
    extract_trace_context,
    SpanAttributes
)
from opentelemetry.trace import SpanKind
import time

tracer = init_tracer("vllm.api_server", "http://localhost:4317")

def handle_inference_request(request_data, headers):
    # Extract incoming trace context
    context = extract_trace_context(headers)

    with tracer.start_as_current_span(
        "inference_request",
        kind=SpanKind.SERVER,
        context=context
    ) as span:
        # Request metadata
        request_id = request_data["request_id"]
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request_id)
        span.set_attribute(SpanAttributes.GEN_AI_RESPONSE_MODEL, "llama-2-7b")
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS,
                          request_data["max_tokens"])
        span.set_attribute(SpanAttributes.GEN_AI_REQUEST_N,
                          request_data["n"])

        start_time = time.time()

        # Queue phase
        with tracer.start_as_current_span("queue"):
            queue_start = time.time()
            # Wait in queue
            queue_time = time.time() - queue_start
            span.set_attribute(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE,
                             queue_time)

        # Generation phase
        with tracer.start_as_current_span("generation"):
            output = generate_tokens(request_data)

        # Set output metrics
        span.set_attribute(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS,
                          output["prompt_tokens"])
        span.set_attribute(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
                          output["completion_tokens"])
        span.set_attribute(SpanAttributes.GEN_AI_LATENCY_E2E,
                          time.time() - start_time)

        return output
</syntaxhighlight>

=== Configure via Environment Variables ===
<syntaxhighlight lang="bash">
# Set OTLP protocol (grpc or http/protobuf)
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc

# Set endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Set service name
export OTEL_SERVICE_NAME=vllm-inference

# Then in Python
from vllm.tracing import init_tracer

tracer = init_tracer(
    "vllm.engine",
    "http://localhost:4317"  # Can be overridden by env var
)
</syntaxhighlight>

=== Graceful Degradation ===
<syntaxhighlight lang="python">
from vllm.tracing import is_otel_available, init_tracer

# Safe initialization
if is_otel_available():
    tracer = init_tracer("vllm.api", "http://localhost:4317")
else:
    tracer = None
    print("Tracing disabled - OpenTelemetry not installed")

# Safe span creation
def traced_function():
    if tracer:
        with tracer.start_as_current_span("function"):
            do_work()
    else:
        do_work()
</syntaxhighlight>

== Semantic Conventions ==

=== Gen AI Span Attributes ===
Based on [https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md OpenTelemetry Semantic Conventions]:

'''Request Attributes:'''
* gen_ai.request.id - Request identifier
* gen_ai.request.max_tokens - Maximum tokens to generate
* gen_ai.request.temperature - Sampling temperature
* gen_ai.request.top_p - Top-p sampling value
* gen_ai.request.n - Number of sequences

'''Response Attributes:'''
* gen_ai.response.model - Model name
* gen_ai.usage.prompt_tokens - Input token count
* gen_ai.usage.completion_tokens - Output token count
* gen_ai.usage.num_sequences - Number of generated sequences

'''Latency Attributes:'''
* gen_ai.latency.e2e - End-to-end latency
* gen_ai.latency.time_in_queue - Queue waiting time
* gen_ai.latency.time_to_first_token - Prefill latency
* gen_ai.latency.time_in_scheduler - Scheduler time
* gen_ai.latency.time_in_model_forward - Model forward pass time
* gen_ai.latency.time_in_model_execute - Total model execution time
* gen_ai.latency.time_in_model_prefill - Prefill stage time
* gen_ai.latency.time_in_model_decode - Decode stage time

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[OpenTelemetry Integration]]
* [[Performance Monitoring]]
* [[Request Metrics]]
* [[Distributed Systems]]
