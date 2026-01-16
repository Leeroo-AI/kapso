# File: `WebAgent/WebWalker/src/rag_system.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 335 |
| Functions | `o1_api`, `gemini_api`, `doubao_api`, `kimi_api`, `wenxin_api`, `main` |
| Imports | aiohttp, asyncio, concurrent, datasets, json, openai, os, requests, tenacity, tqdm, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides API integration functions for running baseline evaluations on the WebWalkerQA dataset using various commercial LLM APIs, enabling comparison with the WebWalker agent approach.

**Mechanism:** Implements async/concurrent API wrappers for multiple LLM providers: (1) `o1_api()` - OpenAI O1-preview with AsyncOpenAI client; (2) `gemini_api()` - Google Gemini 1.5 Pro via HTTP with optional Google Search retrieval tool; (3) `doubao_api()` - ByteDance Doubao via Ark SDK with bot chat; (4) `kimi_api()` - Moonshot Kimi with built-in web search tool support; (5) `wenxin_api()` - Baidu Wenxin via REST API with OAuth token. Each function loads the WebWalkerQA dataset, tracks processed questions for resumption, executes queries with retry logic and rate limiting (semaphores), and writes results to JSONL. The `main()` function dispatches to the appropriate API handler.

**Significance:** Baseline comparison infrastructure that enables evaluating closed-source LLMs (with and without search capabilities) against the WebWalker agent on the same benchmark, supporting ablation studies and performance comparison research.
