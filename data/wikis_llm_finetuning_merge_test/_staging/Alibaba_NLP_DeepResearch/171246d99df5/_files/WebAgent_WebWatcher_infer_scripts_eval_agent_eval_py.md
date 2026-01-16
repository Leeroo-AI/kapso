# File: `WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 481 |
| Classes | `OmniSearch` |
| Imports | PIL, argparse, base64, concurrent, contextlib, datetime, io, json, math, mmrag_r1, ... +7 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main evaluation agent for running vision-language web search inference on multiple benchmarks including HLE, GAIA, LiveVQA, MMSearch, SimpleVQA, and BrowseComp variants.

**Mechanism:** The `OmniSearch` class implements a complete multimodal web search agent:
1. **Initialization:** Sets up OpenAI-compatible client, configures image pixel constraints (min/max), initializes `Qwen_agent` with tools (web_search, VLSearchImage, visit, code_interpreter), and loads processor from HuggingFace
2. **Image Processing:** `process_image()` handles image loading, resizing to meet pixel constraints, RGB conversion, and base64 encoding for API calls
3. **Main Loop (`run_main()`):**
   - Constructs system prompt and user message with image and question
   - Iteratively calls vision-language model via chat completions API
   - Parses `<think>`, `<tool_call>`, and `<answer>` tags from responses
   - Routes tool calls to appropriate handlers: VLSearchImage (visual search with image results), web_search, visit (webpage summaries), code_interpreter (Python execution)
   - Manages conversation state, appending tool responses to message history
   - Enforces maximum step limit (12 steps) before requiring final answer
4. **Inference Execution:**
   - `infer()`: Single sample processing with error handling
   - `infer_with_timeout_retry()`: Adds timeout (300s) and retry logic (max 2 retries)
   - `eval()`: Batch processing with ThreadPoolExecutor (20 workers), progress tracking via tqdm, incremental output to JSONL
5. **Tool Integration:** Leverages `mmrag_r1.llm_agent.qwen_tool_call.Qwen_agent` for unified tool execution

**Significance:** Core inference engine for WebWatcher's multimodal capabilities. Enables systematic evaluation of vision-language web search agents across diverse benchmarks, combining image understanding with web information retrieval. Essential for generating model outputs that are subsequently scored by evaluation scripts.
