# File: `WebAgent/WebResummer/src/tool_search.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 112 |
| Classes | `Search` |
| Imports | concurrent, json, os, qwen_agent, requests, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the web search tool for the agent, enabling batched Google searches via the Serper API.

**Mechanism:** The `Search` class extends `BaseTool` and is registered as "search" tool. It defines a JSON schema accepting an array of query strings. The `google_search` method sends POST requests to Serper's Google search API (google.serper.dev) with retry logic (5 attempts), retrieving top 10 results per query. Results are formatted as markdown with title, URL, date, source, and snippet. The `call` method handles both single queries (string) and batched queries (array), using ThreadPoolExecutor (3 workers) for parallel execution of multiple queries. Batched results are joined with "=======" separators. Requires GOOGLE_SEARCH_KEY environment variable.

**Significance:** Core tool providing the agent's web search capability. The batched search design allows efficient parallel information gathering, enabling the agent to explore multiple query angles simultaneously during web research tasks.
