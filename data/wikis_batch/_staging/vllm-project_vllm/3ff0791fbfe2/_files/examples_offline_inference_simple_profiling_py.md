# File: `examples/offline_inference/simple_profiling.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 52 |
| Functions | `main` |
| Imports | time, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates torch profiler integration

**Mechanism:** Initializes LLM with profiler_config specifying torch profiler and output directory, calls llm.start_profile() before generation, llm.stop_profile() after, then waits for background profiler to write results. Shows basic profiling workflow for performance analysis.

**Significance:** Example showing how to profile vLLM execution using PyTorch profiler.
