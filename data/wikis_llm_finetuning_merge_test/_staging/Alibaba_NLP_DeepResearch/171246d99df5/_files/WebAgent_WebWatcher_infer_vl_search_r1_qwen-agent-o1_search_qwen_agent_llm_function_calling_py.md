# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/function_calling.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 182 |
| Classes | `BaseFnCallModel` |
| Functions | `simulate_response_completion_with_chat`, `validate_num_fncall_results` |
| Imports | abc, copy, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Provides the abstract base class `BaseFnCallModel` that adds function calling (tool use) capabilities to LLM implementations through prompt engineering.

**Mechanism:** `BaseFnCallModel` extends `BaseChatModel` and initializes a function call prompt handler based on `fncall_prompt_type` config (supports 'qwen', 'nous', 'nous_think', 'code' formats). It overrides `_preprocess_messages` to inject function definitions into prompts via the selected prompt formatter and `_postprocess_messages` to parse function calls from model outputs. The `_remove_fncall_messages` method converts function call history into natural language when `function_choice="none"`. The `_chat_with_functions` method delegates to `continue_assistant_response` which uses `simulate_response_completion_with_chat` to handle response continuation by merging the last user and assistant messages. Helper function `validate_num_fncall_results` ensures function calls and results are properly matched.

**Significance:** Core component that enables tool/function calling for models that don't natively support it. By transforming function definitions and calls into prompt templates, this allows the Qwen agent framework to use any text LLM for agentic workflows with tools like search, code execution, and web browsing.
