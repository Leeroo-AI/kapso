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

**Purpose:** Benchmarks long document QA throughput with prefix caching enabled.

**Mechanism:** Samples multiple documents (default: 8) with long prompts (default: 20000 tokens), then replicates each prompt multiple times (default: 2) in different orders (random, tile, or interleave modes). Tests prefix caching effectiveness by measuring execution time across different prompt repetition patterns. The "tile" mode repeats the entire list sequentially (potentially lowest cache hit), "interleave" mode repeats each prompt consecutively (highest cache hit), and "random" mode shuffles prompts. Includes warmup requests before measurement.

**Significance:** Performance validation tool for prefix caching with long-context workloads. Demonstrates the benefits of automatic prefix caching for document QA scenarios where multiple queries reference the same long document. Critical for validating that prefix caching delivers expected performance improvements in real-world long-context applications. Helps optimize caching strategies for RAG and document analysis use cases.
