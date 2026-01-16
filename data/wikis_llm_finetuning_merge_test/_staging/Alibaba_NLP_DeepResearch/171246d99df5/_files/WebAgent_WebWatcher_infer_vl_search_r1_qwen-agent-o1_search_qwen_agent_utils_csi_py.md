# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/utils/csi.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 90 |
| Functions | `generate_random_string`, `get_current_time`, `csi` |
| Imports | copy, datetime, os, random, requests, string |

## Understanding

**Status:** Explored

**Purpose:** Implements Content Safety Inspection (CSI) functionality for checking user-generated content against Alibaba's content moderation service.

**Mechanism:**
- Configures connection to Alibaba's content safety service via environment variables (region, endpoints, business credentials)
- The main `csi()` function takes text content, a document URL, and scene type (default: "search")
- Constructs a JSON payload following Alibaba's security API format with user input, timestamp, and unique request ID
- Makes HTTP POST request to Alibaba's private content moderation endpoint
- Returns moderation result codes: "-1" (no match), "0" (normal/safe), "1" (violation), "3" (suspicious/pending review)
- Helper functions `generate_random_string()` and `get_current_time()` support request ID generation and timestamping

**Significance:** Content moderation utility that integrates with Alibaba Cloud's security infrastructure. Used to filter inappropriate or unsafe content in web search and agent interactions, ensuring compliance with content policies. This is particularly important for the web agent's search functionality to screen both queries and retrieved content.
