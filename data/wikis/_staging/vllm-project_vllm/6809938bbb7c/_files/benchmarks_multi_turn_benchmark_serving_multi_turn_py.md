# File: `benchmarks/multi_turn/benchmark_serving_multi_turn.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1666 |
| Classes | `ConversationSampling`, `ClientArgs`, `RequestArgs`, `BenchmarkArgs`, `ServerResponse`, `RequestStats`, `MetricStats`, `MovingAverage`, `DebugStats` |
| Functions | `nanosec_to_millisec`, `nanosec_to_sec`, `send_request`, `get_short_string`, `get_token_count`, `get_messages_token_count`, `send_turn`, `poisson_sleep`, `... +9 more` |
| Imports | aiohttp, argparse, asyncio, bench_dataset, bench_utils, collections, datetime, enum, http, json, ... +10 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks serving performance for multi-turn conversations with KV cache reuse.

**Mechanism:** Simulates multi-turn conversation workloads, measuring throughput and latency while reusing KV cache across turns. Tests prefix caching and conversation management.

**Significance:** Multi-turn conversations are common in chatbot scenarios. Benchmarking helps optimize KV cache management and prefix caching strategies.
