# File: `WebAgent/WebSailor/src/run_multi_react.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 188 |
| Imports | argparse, concurrent, datetime, json, os, prompt, react_agent, threading, tool_search, tool_visit, ... +1 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main entry point script for running WebSailor agent inference on evaluation datasets, supporting multiple rollouts (iterations) with parallel execution for benchmarking.

**Mechanism:** The script parses command-line arguments (model path, dataset, temperature, workers, rollout count) and loads evaluation data from JSONL files. For each rollout iteration (default 3), it creates a `MultiTurnReactAgent` instance configured with search/visit tools and the system prompt (with current date). Tasks are processed in parallel using `ThreadPoolExecutor` with configurable workers (default 20). Results are written incrementally to iter{n}.jsonl files with thread-safe locking. The script tracks processed queries to support resumption and skips already-completed questions. Supports multiple datasets: GAIA, BrowseComp (zh/en), WebWalker, SimpleQA, TimeQA, xbench-deepsearch, and HLE.

**Significance:** The primary execution script for running WebSailor evaluations at scale. It enables reproducible benchmarking with multiple rollouts and efficient parallel processing of question-answering tasks.
