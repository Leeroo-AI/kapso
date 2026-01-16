# File: `inference/tool_visit.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 256 |
| Classes | `Visit` |
| Functions | `truncate_to_tokens` |
| Imports | concurrent, json, openai, os, prompt, qwen_agent, random, requests, signal, threading, ... +5 more |

## Understanding

**Status:** Explored

**Purpose:** Web page visiting and content extraction tool that fetches URLs, reads their content, and produces goal-directed summaries of the information relevant to the user's research objective.

**Mechanism:** The `Visit` class uses Jina AI's reader service (`r.jina.ai`) to fetch and convert web pages to readable text. After retrieval, content is truncated to 95K tokens and processed through an LLM (via `call_server()`) using the `EXTRACTOR_PROMPT` template to extract: rationale (relevant sections), evidence (key information with full context), and summary (concise paragraph). Supports batch URL processing with 15-minute timeout protection. Includes retry logic for both fetching and summarization, with graceful degradation on failures.

**Significance:** Essential complement to the search tool. While search provides URLs and snippets, this tool performs deep reading of web pages to extract detailed information. The goal-directed summarization ensures relevant content is extracted efficiently, enabling thorough research without overwhelming the context window.
