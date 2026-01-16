# File: `WebAgent/WebSailor/src/tool_search.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 103 |
| Classes | `Search` |
| Imports | concurrent, json, os, qwen_agent, requests, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the web search tool for the WebSailor agent, enabling batched Google searches via the Serper API.

**Mechanism:** The `Search` class is registered as a Qwen Agent tool using `@register_tool("search")`. It accepts a query parameter that can be either a single string or an array of query strings for batch processing. The `google_search()` method sends POST requests to `google.serper.dev/search` with the GOOGLE_SEARCH_KEY from environment variables, requesting top 10 results per query. Results are formatted as markdown with title, link, date, source, and snippet. Batch queries are executed in parallel using `ThreadPoolExecutor` with 3 workers, and results are joined with separator strings. Error handling includes 5 retries for API calls.

**Significance:** Essential tool component that provides web search capability to the WebSailor agent. It enables the agent to discover relevant URLs and information snippets as the first step in web-based question answering.
