# File: `WebAgent/WebResummer/src/prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 169 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines system prompts and templates for the WebResummer agent's web information seeking and conversation summarization capabilities.

**Mechanism:** Contains four key prompts: (1) `EXTRACTOR_PROMPT` - instructs LLM to extract relevant information from webpage content given a user goal, outputting JSON with "rational", "evidence", and "summary" fields; (2) `SYSTEM_PROMPT` - defines the agent as a "Web Information Seeking Master" with principles for persistent searching, repeated verification, and attention to detail, including tool schemas for `search` (batched web queries) and `visit` (webpage summarization); (3) `QUERY_SUMMARY_PROMPT` - for summarizing conversation history without prior context; (4) `QUERY_SUMMARY_PROMPT_LAST` - for incremental summarization building on previous summaries. All summary prompts emphasize extracting only certain, explicitly-stated information.

**Significance:** Foundational prompt engineering component that shapes agent behavior. The prompts establish the agent's reasoning patterns, tool usage conventions (XML-style tags for tool_call/answer), and summarization guidelines that enable effective long-context web research through periodic context compression.
