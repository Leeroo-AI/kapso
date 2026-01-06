# File: `vllm/tracing.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 135 |
| Classes | `SpanAttributes`, `Context`, `BaseSpanAttributes`, `SpanKind`, `Tracer` |
| Functions | `is_otel_available`, `init_tracer`, `get_span_exporter`, `extract_trace_context`, `extract_trace_headers`, `contains_trace_headers`, `log_tracing_disabled_warning` |
| Imports | collections, os, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** OpenTelemetry distributed tracing integration for request tracking.

**Mechanism:** Provides optional OpenTelemetry integration for distributed tracing. `is_otel_available()` checks if OpenTelemetry packages are installed. `init_tracer()` initializes a tracer with OTLP exporter (gRPC or HTTP). `SpanAttributes` defines semantic conventions for AI/ML traces (completion tokens, prompt tokens, temperature, latency metrics). Functions extract trace context from HTTP headers (`traceparent`, `tracestate`) for propagating traces across service boundaries. Gracefully handles missing OpenTelemetry dependencies with fallback stub classes.

**Significance:** Enables production observability for vLLM deployments. Distributed tracing is crucial for understanding request flow in complex systems, identifying bottlenecks, and debugging issues across microservices. The trace context propagation allows vLLM to participate in larger tracing ecosystems. The optional nature (no hard dependency on OpenTelemetry) keeps vLLM lightweight while supporting advanced observability for production users.
