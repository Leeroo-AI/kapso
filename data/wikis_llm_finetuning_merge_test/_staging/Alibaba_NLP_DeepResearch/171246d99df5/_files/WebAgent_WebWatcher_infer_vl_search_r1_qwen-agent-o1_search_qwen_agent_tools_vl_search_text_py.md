# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/vl_search_text.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 328 |
| Classes | `VLSearchText` |
| Functions | `search_cache_decorator` |
| Imports | argparse, atexit, base64, dotenv, functools, hashlib, io, json, logging, os, ... +11 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements text-based image search functionality that retrieves relevant images from the web based on text queries using Alibaba's internal search API.

**Mechanism:** The `VLSearchText` class extends `BaseTool` and provides:
- `search_image_by_text()`: Sends POST requests to Alibaba's Qwen search API with configurable ranking models (including qwen-rerank)
- `search_cache_decorator`: A caching decorator storing results in a JSONL file to avoid duplicate API calls
- Image download pipeline with fallback methods (curl -> wget -> requests)
- Image upload to Alibaba Cloud OSS for hosting retrieved images
- Result parsing that extracts image URLs and captions from search results
- The `call()` method processes text queries, performs image searches, downloads/uploads result images, and returns formatted output

**Significance:** Core search tool for the visual-language (VL) search pipeline in the WebWatcher agent system. Complements `VLSearchImage` by enabling text-to-image search, allowing the agent to find relevant visual content based on textual descriptions. Essential for multimodal research tasks where the agent needs to discover images related to a topic or question.
