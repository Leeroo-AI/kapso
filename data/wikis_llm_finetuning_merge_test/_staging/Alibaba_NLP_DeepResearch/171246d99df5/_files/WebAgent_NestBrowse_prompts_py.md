# File: `WebAgent/NestBrowse/prompts.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 82 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines all prompt templates used by the NestBrowse agent for system instructions, webpage summarization, and tool definitions.

**Mechanism:** Contains four key prompt constants: (1) `SUMMARY_PROMPT` for extracting goal-relevant information from webpages into rational/evidence/summary JSON, (2) `SUMMARY_PROMPT_INCREMENTAL` for incrementally building upon existing summaries when processing sharded content, (3) `SYSTEM_PROMPT_SUMMARY_OURS` for the summarization LLM with output formatting rules, and (4) `SYSTEM_PROMPT_OURS` which defines the browser-use agent persona and declares tool schemas (search, visit, click, fill) in XML format.

**Significance:** Central configuration file that shapes agent behavior. The prompts define the agent's role as a "browser-use agent" for multi-source investigations and establish the structured extraction format for webpage content processing.
