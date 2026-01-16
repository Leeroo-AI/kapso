# File: `WebAgent/WebResummer/src/summary_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 66 |
| Functions | `call_resum_server`, `summarize_conversation` |
| Imports | json, os, prompt, re, requests, time |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utilities for summarizing conversation history using a dedicated ReSum (Resummarization) model server.

**Mechanism:** Two main functions: (1) `call_resum_server` - sends queries to an external ReSum model server (configured via RESUM_TOOL_NAME and RESUM_TOOL_URL environment variables) using OpenAI-compatible chat API, with retry logic (up to 10 attempts), strips `<think>` tags from response and extracts/wraps content in `<summary>` tags; (2) `summarize_conversation` - formats conversation history and question into a prompt using either `QUERY_SUMMARY_PROMPT` (for first summary) or `QUERY_SUMMARY_PROMPT_LAST` (for incremental summaries building on previous ones), then calls the ReSum server. The prompts emphasize extracting only explicitly-stated, certain information without inference.

**Significance:** Critical utility enabling the ReSum mechanism in WebResummer. By periodically summarizing long conversation histories, it allows the agent to handle extensive web research sessions that would otherwise exceed context limits, while preserving essential information for continued reasoning.
