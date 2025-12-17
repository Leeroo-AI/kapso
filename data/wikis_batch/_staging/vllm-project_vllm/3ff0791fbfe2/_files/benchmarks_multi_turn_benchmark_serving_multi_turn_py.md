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

**Purpose:** Comprehensive multi-turn serving benchmark framework

**Mechanism:** Extensive async benchmark framework for multi-turn conversation serving. Supports multiple conversation sampling strategies (random, round-robin, individual-based). ClientArgs/RequestArgs/BenchmarkArgs configure client behavior, request patterns, and benchmark parameters. Sends HTTP requests to OpenAI-compatible chat endpoints, tracks per-turn and per-conversation metrics. Measures TTFT (time to first token), TPOT (time per output token), ITL (inter-token latency), end-to-end latency, throughput. Implements Poisson arrival process for realistic load generation. Collects detailed statistics (MovingAverage, MetricStats) with percentiles. Supports concurrent conversations with configurable think time between turns. Outputs comprehensive JSON results.

**Significance:** Premier tool for evaluating stateful multi-turn serving performance. Critical for understanding prefix caching effectiveness, KV cache management, and scheduling under conversational workloads. Validates that multi-turn optimizations (automatic prefix caching, context caching) provide expected benefits. Essential for production readiness testing of conversational AI deployments. Enables apples-to-apples comparison of different serving configurations. Key for validating vLLM's multi-turn serving capabilities.
