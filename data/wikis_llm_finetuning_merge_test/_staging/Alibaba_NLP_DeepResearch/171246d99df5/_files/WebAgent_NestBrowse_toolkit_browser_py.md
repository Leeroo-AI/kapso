# File: `WebAgent/NestBrowse/toolkit/browser.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 192 |
| Classes | `Visit`, `Click`, `Fill` |
| Imports | aiohttp, asyncio, json, os, re, requests, tiktoken, time, toolkit, typing, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements browser interaction tools (Visit, Click, Fill) that allow the agent to navigate and interact with webpages.

**Mechanism:** Defines three tool classes, each with a `tool_schema` and async `call` method: (1) `Visit` navigates to a URL using `browser_navigate` MCP call and processes the response through `process_response` to extract goal-relevant summaries, (2) `Click` clicks elements by reference ID using `browser_click` and similarly extracts relevant information, (3) `Fill` enters text into input fields using `browser_type`. All tools communicate with a browser MCP server via the client session and use locking for thread-safe operations.

**Significance:** Core interaction layer between the NestBrowse agent and web browsers. These tools enable the agent to perform actual web browsing actions, with built-in content summarization to extract relevant information aligned with user goals.
