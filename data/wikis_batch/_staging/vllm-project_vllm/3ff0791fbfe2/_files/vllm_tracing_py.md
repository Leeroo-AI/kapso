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

**Purpose:** OpenTelemetry tracing integration

**Mechanism:** Provides OpenTelemetry integration for distributed tracing with graceful degradation when OTEL packages are unavailable. Functions include: is_otel_available() (checks OTEL availability), init_tracer() (initializes tracer with OTLP exporter), extract_trace_context() (extracts trace context from HTTP headers), and extract_trace_headers() (filters trace headers). SpanAttributes defines semantic conventions for LLM-specific metrics (latency, token counts, model info). Supports both gRPC and HTTP/protobuf OTLP protocols.

**Significance:** Enables observability for production deployments by integrating with distributed tracing systems (Jaeger, Zipkin, etc.). Critical for debugging latency issues, understanding request flow, and monitoring system health in distributed setups. The optional nature (graceful degradation) ensures vLLM works without OTEL dependencies while enabling advanced observability when needed.
