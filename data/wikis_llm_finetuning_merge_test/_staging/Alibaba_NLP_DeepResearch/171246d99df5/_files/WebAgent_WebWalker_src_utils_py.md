# File: `WebAgent/WebWalker/src/utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 77 |
| Functions | `process_url`, `clean_markdown`, `get_info`, `get_content_between_a_b` |
| Imports | crawl4ai, re, urllib |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utility functions for URL processing, markdown cleaning, and async web crawling used by the WebWalker agent and Streamlit app.

**Mechanism:** Contains four utility functions: (1) `process_url()` - uses `urllib.parse.urljoin()` to resolve relative URLs against a base URL; (2) `clean_markdown()` - removes markdown links, raw URLs, and cleans up whitespace/newlines using regex patterns; (3) `get_info()` - async function using crawl4ai's `AsyncWebCrawler` to fetch webpage content, returning HTML, cleaned markdown, and optionally a base64 screenshot (with 1s wait for rendering); (4) `get_content_between_a_b()` - extracts text between start and end tags, handling multiple occurrences. The crawl4ai integration provides modern async web scraping with screenshot capability.

**Significance:** Foundation utility module that abstracts web content fetching and processing for the WebWalker system. The async crawler with screenshot support enables the visual UI demonstration while the URL/markdown utilities support robust page navigation.
