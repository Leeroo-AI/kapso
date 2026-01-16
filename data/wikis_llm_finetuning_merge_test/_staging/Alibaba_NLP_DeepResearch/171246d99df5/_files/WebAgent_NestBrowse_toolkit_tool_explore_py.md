# File: `WebAgent/NestBrowse/toolkit/tool_explore.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 47 |
| Functions | `process_response` |
| Imports | json5, os, prompts, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Processes raw webpage content by extracting and summarizing goal-relevant information through LLM-based analysis.

**Mechanism:** The `process_response` function handles large webpage content by sharding it into chunks based on token limits (`MAX_SUMMARY_SHARD_LEN`). For the first shard, it uses `SUMMARY_PROMPT`; for subsequent shards, it uses `SUMMARY_PROMPT_INCREMENTAL` to build upon existing evidence and summaries. Each shard is processed via `call_llm` with the summary model, and the response is parsed to extract JSON containing "evidence" and "summary" fields from within `<useful_info>` tags. Returns formatted "Evidence in page" and "Summary" sections.

**Significance:** Critical content processing pipeline that converts raw webpage content into structured, goal-oriented summaries. Enables the agent to handle large webpages by incrementally building comprehensive evidence through multiple LLM calls.
