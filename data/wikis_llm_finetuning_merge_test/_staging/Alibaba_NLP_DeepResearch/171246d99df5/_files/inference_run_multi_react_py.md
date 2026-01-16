# File: `inference/run_multi_react.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 229 |
| Imports | argparse, concurrent, datetime, json, math, os, react_agent, threading, time, tqdm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Main execution script for running the DeepResearch agent on evaluation datasets, supporting parallel execution, multiple rollouts, and dataset splitting for distributed processing.

**Mechanism:** Parses CLI arguments for model path, output directory, dataset file, generation parameters (temperature, top_p, presence_penalty), and parallelization settings. Loads JSON/JSONL datasets and splits them across workers. For each question, assigns ports via round-robin to planning servers (ports 6001-6008). Uses `ThreadPoolExecutor` to run multiple `MultiTurnReactAgent._run()` calls concurrently with configurable max_workers. Supports multiple rollouts per question for ensemble evaluation. Results are written to JSONL files with thread-safe locking, including error handling for timeouts and exceptions.

**Significance:** Production execution harness for the DeepResearch system. Enables large-scale evaluation by running the agent across benchmark datasets with parallelization, checkpointing (skipping already-processed queries), and support for distributed workloads across multiple worker splits.
