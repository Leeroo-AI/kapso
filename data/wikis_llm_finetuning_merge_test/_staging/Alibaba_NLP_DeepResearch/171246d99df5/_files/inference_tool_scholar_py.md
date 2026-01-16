# File: `inference/tool_scholar.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 110 |
| Classes | `Scholar` |
| Imports | concurrent, http, json, os, qwen_agent, requests, typing |

## Understanding

**Status:** Explored

**Purpose:** Google Scholar search tool that enables the DeepResearch agent to retrieve academic publications and scholarly information for research queries.

**Mechanism:** The `Scholar` class connects to Google Scholar via the Serper API (`google.serper.dev`). The `google_scholar_with_serp()` method sends POST requests with search queries and parses results to extract: title, PDF URL, publication info, publication year, citation count, and snippet. Supports batch queries through ThreadPoolExecutor (max 3 workers) for parallel search execution. Results are formatted as numbered markdown entries with links and metadata.

**Significance:** Specialized search tool for academic research. Complements the general web search tool by providing access to scholarly literature, enabling the agent to find peer-reviewed sources, research papers, and authoritative academic content for in-depth research questions.
