# File: `WebAgent/ParallelMuse/compressed_reasoning_aggregation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 295 |
| Functions | `today_date`, `get_llm_response`, `read_jsonl`, `write_jsonl`, `cluster_by_question`, `construct_interaction_from_record`, `call_state_report`, `call_info_integrate`, `... +2 more` |
| Imports | asyncio, collections, datetime, json, openai, os, random, re, time, tqdm, ... +1 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Aggregates multiple parallel reasoning trajectories into a single final answer by generating reports from each trajectory and integrating them through LLM-based consensus analysis.

**Mechanism:** Implements a two-stage pipeline: (1) `call_state_report` converts each trajectory into a structured "problem-solving report" containing solution planning, methods, and final reasoning using `REPORT_PROMPT`; (2) `call_info_integrate` combines multiple reports using `INTEGRATE_PROMPT`, which instructs the LLM to critically evaluate consistency across reports and select the most trustworthy answer. The `call_converge` function orchestrates this process for trajectory groups clustered by question. Results are parsed from `<report>` and `<answer>` XML tags.

**Significance:** Key component of the ParallelMuse parallel sampling strategy. Enables the system to run multiple independent reasoning paths and then intelligently aggregate them, leveraging consensus and cross-validation to improve answer quality and reliability.
