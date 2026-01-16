# File: `WebAgent/WebSailor/src/tool_visit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 220 |
| Classes | `Visit` |
| Imports | concurrent, json, openai, os, prompt, qwen_agent, random, requests, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the webpage visit tool for the WebSailor agent, fetching and summarizing webpage content with goal-directed extraction.

**Mechanism:** The `Visit` class is registered as a Qwen Agent tool using `@register_tool('visit')`. It accepts URL(s) and a goal parameter. The `jina_readpage()` method fetches webpage content via Jina Reader API (`r.jina.ai/`), with retry logic. Content is truncated to WEBCONTENT_MAXLENGTH (150K chars default). The `readpage()` method then calls a local LLM server (Qwen2.5-72B at port 6002) with the EXTRACTOR_PROMPT to extract goal-relevant information in JSON format with "rational", "evidence", and "summary" fields. Multiple URLs are processed in parallel using ThreadPoolExecutor. The tool handles various failure modes including parsing errors and content truncation with progressive retry.

**Significance:** Critical tool component that enables deep webpage exploration in WebSailor. By combining content fetching with LLM-based summarization, it extracts goal-relevant information efficiently within token constraints.
