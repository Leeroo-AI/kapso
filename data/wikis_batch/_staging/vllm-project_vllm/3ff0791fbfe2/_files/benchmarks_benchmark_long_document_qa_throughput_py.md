# File: `benchmarks/benchmark_long_document_qa_throughput.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 202 |
| Functions | `test_long_document_qa`, `repeat_prompts`, `main`, `create_argument_parser` |
| Imports | dataclasses, random, time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test prefix caching throughput

**Mechanism:** Benchmarks long document QA performance with automatic prefix caching. Creates documents by repeating tokens, then duplicates and shuffles prompts in various patterns (random/tile/interleave) to test cache hit rates. Supports warmup phase and measures total execution time. Configurable document length (~20K tokens), repeat counts, and output lengths.

**Significance:** Essential for evaluating prefix caching effectiveness on long-context workloads. Demonstrates performance benefits when serving multiple requests with shared prefixes, simulating real-world scenarios like RAG systems querying the same large documents repeatedly.
