# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/parallel_run.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 133 |
| Functions | `update_call_args`, `llm_api_wrapper`, `llm_api_generate` |
| Imports | concurrent, copy, functools, json, logger, multiprocessing, openai_style_api_client, os, sys, time, ... +3 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Provides batch processing infrastructure for running multiple LLM API calls in parallel with progress tracking, caching, and result persistence.

**Mechanism:** Key components:
- `update_call_args(data, call_args)`: Prepares call arguments by merging data-level and command-line args, constructs messages array from prompt/system fields
- `llm_api_wrapper(api_call, data)`: Wrapper that executes a single API call, extracts generated content from response choices, handles exceptions with sleep delays
- `llm_api_generate(args, api_call)`: Main orchestrator that:
  1. Loads input data from file (via `load_file2list`)
  2. Supports local caching by tracking processed UUIDs to avoid reprocessing
  3. Uses `ThreadPoolExecutor` for parallel execution with configurable worker count
  4. Displays progress via `tqdm` progress bar
  5. Uses multiprocessing Lock for thread-safe file writes
  6. Writes results as JSONL with flush after each completion
- Constants: `LLM_CALL_SLEEP = 2`, global `LOCK` for file synchronization

**Significance:** Essential batch processing utility enabling high-throughput LLM inference workloads. Supports the evaluation and data generation pipelines by allowing parallel API calls with resumable processing (via UUID-based caching), making it practical to process large datasets efficiently.
