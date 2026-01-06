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

**Purpose:** Provides async request functions for benchmarking multiple LLM serving backends (vLLM, TGI, TensorRT-LLM, DeepSpeed-MII, OpenAI, etc.).

**Mechanism:** Implements backend-specific async request handlers that send prompts to different serving frameworks via their APIs (HTTP streaming), measure latency metrics (TTFT, inter-token latency), and collect generated outputs. Each async function handles the specific API format and streaming protocol of its target backend. Includes tokenizer utilities for model loading and prompt tokenization.

**Significance:** Core benchmark infrastructure file that enables fair cross-backend performance comparisons. Used by benchmark_serving.py and other serving benchmarks to abstract away backend-specific API details. Supports both completions and chat completions endpoints, as well as audio transcription/translation for multimodal models.
