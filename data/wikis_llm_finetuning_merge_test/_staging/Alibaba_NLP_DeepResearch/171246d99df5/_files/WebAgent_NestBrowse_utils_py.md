# File: `WebAgent/NestBrowse/utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 81 |
| Functions | `call_llm`, `read_jsonl`, `count_tokens` |
| Imports | aiohttp, ast, json, openai, os, random |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utility functions for LLM communication, file I/O, and token counting used throughout NestBrowse.

**Mechanism:** Contains three key functions: (1) `call_llm` - async function that calls OpenAI-compatible APIs with retry logic (up to 10 retries), supports two modes ('agent' and 'summary') with different API configurations, uses semaphore-based concurrency control, and handles timeout errors by halving max tokens; (2) `read_jsonl` - reads JSONL files and returns a list of JSON objects; (3) `count_tokens` - counts tokens using a HuggingFace tokenizer, supporting both raw strings and chat message lists via `apply_chat_template`.

**Significance:** Essential utility layer that abstracts LLM API interactions and provides common functions used across the NestBrowse agent. The separation of agent vs summary modes allows using different models/endpoints for reasoning and content summarization.
