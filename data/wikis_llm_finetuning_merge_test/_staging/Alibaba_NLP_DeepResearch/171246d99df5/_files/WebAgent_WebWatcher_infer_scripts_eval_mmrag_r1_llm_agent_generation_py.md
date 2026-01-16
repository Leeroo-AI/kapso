# File: `WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/llm_agent/generation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 730 |
| Classes | `GenerationConfig`, `LLMGenerationManager` |
| Functions | `process_image` |
| Imports | PIL, collections, concurrent, dataclasses, json, numpy, os, qwen_tool_call, re, requests, ... +7 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Orchestrates multi-turn LLM generation with tool calling for multi-modal retrieval-augmented generation (RAG) tasks, handling text and image inputs/outputs.

**Mechanism:** The `LLMGenerationManager` class manages the complete generation loop: (1) Processes multi-modal inputs (images resized to pixel bounds, text tokenized). (2) Runs iterative generation with `run_llm_loop` - generating responses, parsing tool calls (`<tool_call>`, `<answer>` tags), executing tools via Qwen_agent, and processing observations. (3) Handles GPU padding for multi-GPU inference. (4) Manages rolling state updates with attention masks, position IDs, and multi-modal data concatenation. (5) Executes tool calls in parallel using ThreadPoolExecutor for web_search, VLSearchImage, and visit tools. (6) `GenerationConfig` dataclass controls max_turns, prompt lengths, and response limits.

**Significance:** Core orchestration component of the WebWatcher system. Implements the agentic loop that allows the LLM to iteratively gather information through tool use before providing final answers. Integrates with verl (verification library) for reinforcement learning and supports both training and validation modes.
