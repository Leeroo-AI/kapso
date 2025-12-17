# File: `benchmarks/backend_request_func.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 657 |
| Classes | `RequestFuncInput`, `RequestFuncOutput` |
| Functions | `async_request_tgi`, `async_request_trt_llm`, `async_request_deepspeed_mii`, `async_request_openai_completions`, `async_request_openai_chat_completions`, `async_request_openai_audio`, `get_model`, `get_tokenizer` |
| Imports | aiohttp, dataclasses, huggingface_hub, io, json, os, sys, time, tqdm, traceback, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unified async HTTP request handlers for multiple LLM serving backends

**Mechanism:** Provides async request functions for different serving backends (TGI, TRT-LLM, DeepSpeed-MII, OpenAI-compatible APIs) with streaming support. Uses aiohttp for async HTTP requests, handles different API formats (completions, chat completions, audio), tracks timing metrics (TTFT, ITL, TPOT), and includes model/tokenizer loading utilities. Implements backend-agnostic request/response data classes.

**Significance:** Critical utility for benchmarking tools that need to test against multiple LLM serving backends. Enables fair performance comparisons by providing standardized request interfaces with consistent timing measurements across different backend implementations (vLLM, TGI, TensorRT-LLM, etc.).
