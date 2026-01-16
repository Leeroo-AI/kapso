# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/readpage.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 195 |
| Classes | `Visit` |
| Functions | `jina_readpage`, `aidata_readpage` |
| Imports | asyncio, json, os, qwen_agent, random, requests, uniform_eval |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Alternative implementation of the webpage visit tool that fetches content and summarizes it using an LLM, with support for multiple backend services.

**Mechanism:** Provides two standalone functions and a `Visit` class:
- `jina_readpage(url)`: Fetches webpage content via Jina AI reader API with bearer token authentication
- `aidata_readpage(url)`: Uses Alibaba's AIData service through TopApiClient SDK, with configurable cache-only mode via NLP_WEB_SEARCH_ONLY_CACHE env var
- `Visit` class (registered as 'visit' tool):
  - Takes url and goal parameters
  - `readpage()`: Fetches content and passes to LLM for extraction
  - `llm()`: Async method calling a configurable judge model (JUDGE_MODEL env var) with JSON schema enforcement for rational/evidence/summary output
  - 10 retry attempts for robust operation
- Uses extractor_prompt template for structured information extraction

**Significance:** Simpler variant of the visit tool compared to jialong_visit.py. Provides flexibility in deployment scenarios where different backend services may be preferred. The modular design allows swapping between Jina and AIData services via environment configuration.
