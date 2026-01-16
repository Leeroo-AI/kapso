# File: `WebAgent/WebDancer/demos/tools/private/visit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 173 |
| Classes | `Visit` |
| Functions | `jina_readpage` |
| Imports | concurrent, json, openai, os, qwen_agent, requests, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the `Visit` tool for fetching and summarizing webpage content with goal-oriented extraction.

**Mechanism:** Extends `BaseTool` and registers as 'visit' tool. Key components: (1) `jina_readpage()` - fetches webpage content via Jina Reader API (r.jina.ai) with retries, (2) `call()` - accepts 'url' (string or array) and 'goal' parameters, uses ThreadPoolExecutor for parallel URL processing, (3) `readpage()` - fetches page via Jina, then calls `llm()` with an extraction prompt to summarize content relative to the user's goal, (4) `llm()` - uses qwen2.5-72b-instruct via DashScope OpenAI-compatible API to extract evidence and summary in JSON format. The extraction prompt guides the LLM to scan content, identify relevant sections, and produce structured output with rational, evidence, and summary fields. Requires JINA_API_KEY and DASHSCOPE_API_KEY environment variables.

**Significance:** Core tool for WebDancer's deep information retrieval. Enables the agent to not just find URLs but intelligently extract and summarize relevant information from webpages based on the search goal.
