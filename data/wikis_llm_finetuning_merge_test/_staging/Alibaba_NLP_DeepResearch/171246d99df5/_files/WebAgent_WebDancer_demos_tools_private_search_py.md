# File: `WebAgent/WebDancer/demos/tools/private/search.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 99 |
| Classes | `Search` |
| Imports | concurrent, json, os, qwen_agent, requests, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the `Search` tool for performing batched web searches via Google's Serper API.

**Mechanism:** Extends `BaseTool` from qwen_agent and registers as 'search' tool. Key features: (1) `call()` - accepts JSON params with 'query' field (string or array), limits to MAX_MULTIQUERY_NUM queries (default 3), uses ThreadPoolExecutor for parallel search execution, (2) `google_search()` - makes POST request to serper.dev API with query, parses 'organic' results extracting title, link, date, source, and snippet, formats as numbered markdown list with links. Returns formatted results like "A Google search for 'X' found N results:" followed by numbered entries. Requires GOOGLE_SEARCH_KEY environment variable.

**Significance:** Core tool for WebDancer's web information seeking capability. Enables the agent to query the web and retrieve relevant search results that can then be visited for detailed information.
