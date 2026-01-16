# File: `WebAgent/WebResummer/src/main.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 164 |
| Imports | argparse, concurrent, json, os, prompt, react_agent, threading, tool_search, tool_visit, tqdm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main entry point for running the WebResummer agent to perform web research tasks across multiple rollouts on evaluation datasets.

**Mechanism:** Parses command-line arguments for model path, output directory, dataset, temperature, parallelism settings, and rollout count. Loads evaluation data from JSONL/JSON files and creates a `MultiTurnReactAgent` with search/visit tools and a system prompt. Uses `ProcessPoolExecutor` for parallel task execution across multiple rollouts, with each rollout producing an `iter{N}.jsonl` output file. Implements resume capability by tracking processed questions per rollout. Thread-safe file writing is ensured via per-rollout locks. Handles errors gracefully by logging failed tasks with error information.

**Significance:** Core orchestration component that ties together the agent, tools, and evaluation workflow. Enables scalable benchmarking with multiple rollouts for computing Pass@k metrics, supporting both new runs and continuation of interrupted experiments.
