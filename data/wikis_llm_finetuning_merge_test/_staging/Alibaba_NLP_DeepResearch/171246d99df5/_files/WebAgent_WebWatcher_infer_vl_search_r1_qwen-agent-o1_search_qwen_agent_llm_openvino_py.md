# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/openvino.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 148 |
| Classes | `OpenVINO`, `StopSequenceCriteria` |
| Imports | copy, pprint, qwen_agent, threading, typing |

## Understanding

**Status:** Explored

**Purpose:** Provides local LLM inference using Intel's OpenVINO runtime, enabling CPU-optimized execution of quantized Qwen models without requiring a GPU.

**Mechanism:** The `OpenVINO` class extends `BaseFnCallModel` and requires an `ov_model_dir` config pointing to an OpenVINO-exported model. It initializes `OVModelForCausalLM` from optimum-intel with optional device selection and custom OpenVINO config. The class implements custom stopping criteria via `StopSequenceCriteria` that monitors for stop sequences in decoded output. `_chat_stream` runs generation in a background thread using HuggingFace's `TextIteratorStreamer` for token-by-token output, while `_chat_no_stream` performs synchronous generation. Both methods use `build_text_completion_prompt` to format messages into the expected prompt template before tokenization.

**Significance:** Enables edge/offline deployment of the Qwen agent framework on CPU-only hardware. By leveraging OpenVINO's INT4/INT8 quantization and CPU optimizations, this allows running AI agents on standard servers or laptops without GPU requirements, making the framework accessible for resource-constrained or air-gapped environments.
