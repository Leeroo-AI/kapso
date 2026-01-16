# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/visit.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 239 |
| Classes | `Visit` |
| Imports | concurrent, json, openai, os, qwen_agent, random, requests, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a webpage visiting tool using Jina reader service and a local vLLM server for content summarization, designed for single-node deployment scenarios.

**Mechanism:** The `Visit` class (registered as 'visit' tool) provides:
- `call()`: Entry point accepting url (string or array) and goal parameters, with parallel processing via ThreadPoolExecutor (3 workers)
- `jina_readpage(url)`: Fetches webpage via Jina AI reader (https://r.jina.ai/) with 3 retries and 10s timeout
- `call_server(msgs)`: Calls local vLLM OpenAI-compatible server (http://127.0.0.1:6002/v1) for summarization
  - Uses SUMMERY_MODEL_PATH env var for model selection
  - Implements JSON extraction fallback for malformed responses
  - 10 retry attempts with temperature 0.7
- `readpage(url, goal)`: Orchestrates the full pipeline:
  1. Fetches content via Jina (up to WEBCONTENT_MAXLENGTH chars)
  2. Sends to LLM with EXTRACTOR_PROMPT for rational/evidence/summary extraction
  3. Implements progressive truncation (70% -> 25000 chars) for oversized content
  4. Parses JSON response with retry logic

**Significance:** Lightweight deployment variant of the visit tool optimized for local inference setups. Uses vLLM server instead of remote API calls, making it suitable for offline or resource-constrained environments. The local server approach reduces latency and API costs while maintaining the same extraction quality.
