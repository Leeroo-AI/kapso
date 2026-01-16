# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/nlp_web_search.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 137 |
| Classes | `WebSearch` |
| Imports | atexit, dataclasses, datetime, functools, json, os, qwen_agent, random, re, requests, ... +3 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a web search tool that integrates with Google's Serper API to retrieve search results for the agent's information-seeking tasks.

**Mechanism:** The `WebSearch` class (registered as 'web_search' tool) implements:
- `google_search(query)`: Calls Serper API (https://google.serper.dev/search) with the query, requesting top 10 results
- Parses organic search results to extract title, link, date, source, and snippet
- Formats results as numbered markdown-style entries with clickable links
- Supports batch queries via ThreadPoolExecutor for parallel processing
- Uses retry logic (5 attempts) for API resilience
- Configuration via environment variables: TEXT_SEARCH_KEY (API key), SEARCH_ENGINE, SEARCH_STRATEGY, MAX_CHAR
- Includes knowledge snippet templates for formatting retrieved information

**Significance:** Essential tool that provides the web search capability for the DeepResearch agent. Enables the agent to discover relevant URLs and snippets before visiting pages. The structured output format makes it easy for the agent to identify and select URLs for deeper exploration via the visit tool.
