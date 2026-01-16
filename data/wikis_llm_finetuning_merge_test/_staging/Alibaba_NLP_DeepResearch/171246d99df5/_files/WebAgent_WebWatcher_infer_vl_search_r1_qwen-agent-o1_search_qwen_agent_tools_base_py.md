# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 202 |
| Classes | `ToolServiceError`, `BaseTool`, `BaseToolWithFileAccess` |
| Functions | `register_tool`, `is_tool_schema` |
| Imports | abc, json, os, qwen_agent, typing |

## Understanding

**Status:** Explored

**Purpose:** Provides the foundational base classes and registration infrastructure for all tools in the Qwen agent framework, implementing an OpenAI-compatible tool calling interface.

**Mechanism:** Implements three key components: (1) `TOOL_REGISTRY` - a global dictionary that maps tool names to their classes; (2) `@register_tool(name)` decorator that registers tool classes into the registry with conflict detection; (3) `BaseTool` - an abstract base class requiring subclasses to implement `call()` method, with built-in JSON parameter validation via jsonschema, automatic argument format detection (Chinese/English), and tool schema generation compatible with OpenAI function calling; (4) `BaseToolWithFileAccess` - extends BaseTool with a working directory and automatic remote file downloading capability. Also includes `is_tool_schema()` to validate OpenAI-compatible JSON schemas and `ToolServiceError` for error handling.

**Significance:** Foundational infrastructure file that defines the tool abstraction layer for the entire agent framework. All tools (CodeInterpreter, DocParser, Storage, etc.) inherit from these base classes. The registration pattern enables dynamic tool discovery and instantiation, while the OpenAI-compatible schema design ensures interoperability with standard LLM function calling interfaces.
