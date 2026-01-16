# File: `WebAgent/WebResummer/src/tool_visit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 240 |
| Classes | `Visit` |
| Functions | `truncate_to_tokens` |
| Imports | json, openai, os, prompt, qwen_agent, requests, tiktoken, time, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the webpage visit tool for the agent, fetching and intelligently summarizing webpage content based on a specified goal.

**Mechanism:** The `Visit` class extends `BaseTool` and is registered as "visit" tool. It accepts URL(s) and a goal string. The workflow: (1) `jina_readpage` fetches webpage content via Jina AI's reader service (r.jina.ai) with retry logic; (2) content is truncated to 95K tokens using tiktoken; (3) an EXTRACTOR_PROMPT is sent to an LLM summarization server (configured via SUMMARY_URL, SUMMARY_API_KEY, SUMMARY_MODEL_NAME env vars) to extract goal-relevant information; (4) response is parsed as JSON with "evidence" and "summary" fields. Supports batched URL processing with a 15-minute timeout. Multiple retry strategies handle summarization failures by progressively truncating content.

**Significance:** Essential tool enabling deep web content analysis. Unlike simple web scraping, it provides intelligent, goal-directed content extraction and summarization, allowing the agent to efficiently process and understand webpage information relevant to the research task.
