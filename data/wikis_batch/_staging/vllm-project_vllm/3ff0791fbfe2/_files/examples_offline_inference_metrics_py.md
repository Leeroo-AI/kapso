# File: `examples/offline_inference/metrics.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 50 |
| Functions | `main` |
| Imports | vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates metrics collection and display

**Mechanism:** Generates text with disable_log_stats=False, calls llm.get_metrics() after generation, and iterates through returned metric objects (Gauge, Counter, Vector, Histogram) to display their values. Shows how to access vLLM's internal performance metrics.

**Significance:** Example demonstrating programmatic access to vLLM metrics for monitoring and observability.
