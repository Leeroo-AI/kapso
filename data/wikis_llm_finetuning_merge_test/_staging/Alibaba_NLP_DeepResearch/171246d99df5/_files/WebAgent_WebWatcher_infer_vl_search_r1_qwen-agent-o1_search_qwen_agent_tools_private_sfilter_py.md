# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/private/sfilter.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 159 |
| Functions | `multi_call_sfilter`, `call_func_in_threads`, `split_sentence_func`, `sfilter` |
| Imports | concurrent, json, os, qwen_agent, re, requests, tqdm, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a sentence-level filtering mechanism that compresses web content by extracting only query-relevant sentences using an LLM model.

**Mechanism:** Key functions include:
- `multi_call_sfilter(query, ctxs)`: Batch processes multiple contexts in parallel for relevance filtering
- `call_func_in_threads(func, args_list)`: Generic thread pool executor wrapper with configurable max workers (NLP_WEB_SEARCH_SFILTER_MAX_WORKERS)
- `split_sentence_func(text)`: Sophisticated sentence splitter handling Chinese/English punctuation, abbreviations (e.g., "Dr.", "U.S.A."), and edge cases
- `sfilter(query, ctx)`: Core filtering function that:
  1. Skips if no web_main_body or it's shorter than snippet
  2. Splits combined snippet+web_main_body into numbered sentences
  3. Calls "pre-sfilter-0.1-server" model via DashScope API to identify relevant sentence indices
  4. Returns compressed text containing only relevant sentences, or original snippet if none found
- Uses Chinese prompt template for extraction instruction

**Significance:** Important optimization component for reducing context length before LLM processing. By filtering out irrelevant sentences from web content, it improves both efficiency (fewer tokens) and quality (focused context). The parallel processing capability enables efficient batch filtering of multiple search results.
