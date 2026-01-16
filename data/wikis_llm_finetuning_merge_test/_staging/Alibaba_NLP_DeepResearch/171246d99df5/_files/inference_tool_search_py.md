# File: `inference/tool_search.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 131 |
| Classes | `Search` |
| Imports | asyncio, concurrent, http, json, os, qwen_agent, requests, typing, uuid |

## Understanding

**Status:** Explored

**Purpose:** Web search tool that enables the DeepResearch agent to perform Google web searches and retrieve current information from the internet.

**Mechanism:** The `Search` class uses the Serper API (`google.serper.dev`) to execute Google searches. The `google_search_with_serp()` method detects Chinese characters in queries to adjust locale settings (China/Chinese vs US/English). Sends POST requests with query payload and parses "organic" results to extract: title, URL, publication date, source, and snippet. Supports batched queries by iterating through query arrays. Results are formatted as numbered markdown entries with retry logic (5 attempts) for reliability.

**Significance:** Primary information retrieval tool for the agent. Enables access to current web content, news, documentation, and general internet information. Essential for research tasks requiring up-to-date information beyond the model's training data.
