# File: `WebAgent/ParallelMuse/functionality_specified_partial_rollout.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 526 |
| Functions | `today_date`, `read_jsonl`, `get_next_base_url`, `call_llm`, `call_tool`, `count_tokens`, `get_initial_rollouts`, `rollout_single_traj`, `... +2 more` |
| Imports | aiohttp, argparse, asyncio, collections, copy, datetime, json, json5, math, numpy, ... +9 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements functionality-specified partial rollout sampling, a strategy that identifies high-uncertainty decision points in agent trajectories and branches from those points to explore alternative reasoning paths.

**Mechanism:** The system operates in two modes: (1) full trajectory rollout ('none' mode) - runs complete agent trajectories from scratch; (2) partial sampling mode - uses perplexity (PPL) metrics to identify uncertain steps. The `call_llm` function collects logprobs and computes entropy/PPL for thinking and tool-call segments. `branch_high_uncertainty_steps` identifies top-k highest uncertainty steps based on thinking PPL, tool-call PPL, or mixed modes. `rollout_single_traj` executes agent trajectories with tool calls (search, visit). The system supports configurable sampling budgets, initial rollout counts, and multi-round partial sampling.

**Significance:** Core innovation of ParallelMuse for efficient parallel exploration. Instead of running many independent full trajectories, it intelligently allocates compute by branching from uncertain decision points, enabling more diverse solution paths while staying within compute budgets.
