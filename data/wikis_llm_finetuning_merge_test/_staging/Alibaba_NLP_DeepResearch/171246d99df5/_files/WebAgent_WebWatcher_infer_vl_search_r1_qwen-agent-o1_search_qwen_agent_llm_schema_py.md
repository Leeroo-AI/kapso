# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/llm/schema.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 142 |
| Classes | `BaseModelCompatibleDict`, `FunctionCall`, `ContentItem`, `Message` |
| Imports | pydantic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the core data structures for messages, content items, and function calls used throughout the qwen-agent LLM framework.

**Mechanism:** Built on Pydantic with four main classes: (1) `BaseModelCompatibleDict` - a base class adding dict-like access (`__getitem__`, `__setitem__`, `get`) to Pydantic models and default `exclude_none=True` serialization, (2) `FunctionCall` - represents tool/function calls with `name` and `arguments` fields, (3) `ContentItem` - represents a single content piece with mutually exclusive fields (text, image, file, audio, video) enforced by a validator, includes `get_type_and_value()` and `type`/`value` properties, (4) `Message` - the main message type with `role` (validated to user/assistant/system/function), `content` (string or list of ContentItems), optional `name`, `function_call`, and `extra` fields. Also defines constants: role types (SYSTEM, USER, ASSISTANT, FUNCTION), content types (FILE, IMAGE, AUDIO, VIDEO), and DEFAULT_SYSTEM_MESSAGE.

**Significance:** Foundational schema module that provides the type-safe message structures used by all LLM implementations in the framework. These schemas ensure consistent message handling across DashScope, OpenAI, and other backends, and support both text-only and multimodal conversations.
