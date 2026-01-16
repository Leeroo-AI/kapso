# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/fncall_prompts/base_fncall_prompt.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 72 |
| Classes | `BaseFnCallPrompt` |
| Imports | qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Defines the abstract base class `BaseFnCallPrompt` that establishes the interface for all function calling prompt format implementations.

**Mechanism:** The class provides three key methods: (1) `preprocess_fncall_messages()` - a static method that transforms structured function call messages into plaintext format with embedded function calling prompts (raises NotImplementedError as abstract); (2) `postprocess_fncall_messages()` - a static method that parses plaintext model output back into structured FunctionCall messages (raises NotImplementedError as abstract); (3) `format_plaintext_train_samples()` - a concrete method that formats training samples by detecting language (Chinese/English), validating parallel function call settings, converting messages to multimodal format, and then calling preprocess to generate plaintext training data.

**Significance:** Core architectural component that enables polymorphic handling of different function calling formats (Qwen, Nous, Code styles). This abstraction is essential for the qwen-agent framework to support multiple LLM backends that expect different function calling syntax, making the agent system model-agnostic.
