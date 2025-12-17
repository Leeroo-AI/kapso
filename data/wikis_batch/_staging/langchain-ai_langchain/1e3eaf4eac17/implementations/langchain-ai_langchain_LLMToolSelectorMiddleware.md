# LLMToolSelectorMiddleware

## Metadata
- **Source:** `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/tool_selection.py`
- **Domains:** agents, middleware, tool-selection, llm
- **Last Updated:** 2025-12-17

## Overview

`LLMToolSelectorMiddleware` is an agent middleware component that uses a large language model to intelligently select the most relevant subset of tools before invoking the main agent model. When an agent has access to many tools, this middleware reduces token usage and improves focus by filtering down to only the most pertinent tools for a given user query.

### Description

The middleware operates by intercepting model calls through the `wrap_model_call` and `awrap_model_call` methods. Before the main model is invoked, it:

1. Extracts available tools from the request
2. Creates a dynamic structured output schema with tool names as Literal types
3. Invokes a selection model (either dedicated or the main model) with the user's last message
4. Filters the tool list based on the selection results
5. Passes the filtered request to the main model

The selection process uses structured output to ensure valid tool names are returned. Tools can be limited to a maximum count, and certain tools can be marked as always included regardless of selection.

### Usage

The middleware is designed to be used with the LangChain agent framework and is particularly valuable when dealing with agents that have access to large tool catalogs:

```python
from langchain.agents.middleware import LLMToolSelectorMiddleware

# Basic usage: limit to 3 most relevant tools
middleware = LLMToolSelectorMiddleware(max_tools=3)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[tool1, tool2, tool3, tool4, tool5],
    middleware=[middleware],
)

# Use a different model for selection
middleware = LLMToolSelectorMiddleware(
    model="openai:gpt-4o-mini",
    max_tools=2
)

# Always include specific tools
middleware = LLMToolSelectorMiddleware(
    max_tools=3,
    always_include=["calculator", "search"]
)
```

## Code Reference

### Source Location
- **File:** `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/tool_selection.py`
- **Lines:** 88-320
- **Class:** `LLMToolSelectorMiddleware`

### Signature
```python
class LLMToolSelectorMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        model: str | BaseChatModel | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
    ) -> None:
        ...
```

### Import Statement
```python
from langchain.agents.middleware import LLMToolSelectorMiddleware
```

## I/O Contract

### Initialization Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `str \| BaseChatModel \| None` | No | `None` | Model to use for selection. If not provided, uses the agent's main model. Can be a model identifier string or `BaseChatModel` instance. |
| `system_prompt` | `str` | No | `DEFAULT_SYSTEM_PROMPT` | Instructions for the selection model. Default instructs the model to select the most relevant tools. |
| `max_tools` | `int \| None` | No | `None` | Maximum number of tools to select. If the model selects more, only the first `max_tools` will be used. If not specified, there is no limit. |
| `always_include` | `list[str] \| None` | No | `None` | Tool names to always include regardless of selection. These do not count against the `max_tools` limit. |

### Method: `wrap_model_call`

Synchronous method that filters tools before model invocation.

| Aspect | Details |
|--------|---------|
| **Input** | `request: ModelRequest` - The original model request with tools<br>`handler: Callable[[ModelRequest], ModelResponse]` - The handler to invoke |
| **Output** | `ModelCallResult` - Result from the handler with filtered tools |
| **Side Effects** | Makes an additional LLM call to select tools if selection is needed |

### Method: `awrap_model_call`

Asynchronous version of `wrap_model_call`.

| Aspect | Details |
|--------|---------|
| **Input** | `request: ModelRequest` - The original model request with tools<br>`handler: Callable[[ModelRequest], Awaitable[ModelResponse]]` - The async handler |
| **Output** | `ModelCallResult` - Result from the handler with filtered tools |
| **Side Effects** | Makes an additional async LLM call to select tools if selection is needed |

### Processing Logic

The middleware follows this flow:

1. **Preparation Phase** (`_prepare_selection_request`):
   - Returns `None` if no tools available or all tools are in `always_include`
   - Validates that `always_include` tools exist in the request
   - Separates tools available for selection from always-included tools
   - Extracts the last user message from conversation history
   - Creates a `_SelectionRequest` with prepared inputs

2. **Selection Phase**:
   - Creates dynamic TypeAdapter with Literal union of available tool names
   - Invokes selection model with system prompt and user message
   - Returns structured output as a dictionary with selected tool names

3. **Filtering Phase** (`_process_selection_response`):
   - Validates selected tool names are in the valid tool list
   - Applies `max_tools` limit (takes first N selected tools)
   - Combines selected tools with always-included tools
   - Preserves any provider-specific tool dictionaries
   - Returns modified `ModelRequest` with filtered tools

## Usage Examples

### Example 1: Basic Tool Limiting

```python
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain import create_agent

# Create middleware that limits to top 3 tools
middleware = LLMToolSelectorMiddleware(max_tools=3)

# Create agent with many tools
agent = create_agent(
    model="openai:gpt-4o",
    tools=[
        search_tool,
        calculator_tool,
        weather_tool,
        stock_tool,
        news_tool,
        translation_tool,
    ],
    middleware=[middleware],
)

# The middleware will select only the 3 most relevant tools per query
response = agent.invoke("What is the weather in San Francisco?")
# Only weather-related tools will be passed to the main model
```

### Example 2: Using a Smaller Selection Model

```python
# Use a faster, cheaper model for tool selection
middleware = LLMToolSelectorMiddleware(
    model="openai:gpt-4o-mini",  # Smaller model for selection
    max_tools=2,
    system_prompt="Select the 2 most relevant tools for this query."
)

agent = create_agent(
    model="openai:gpt-4o",  # Main model for reasoning
    tools=my_tools,
    middleware=[middleware],
)
```

### Example 3: Always-Include Tools

```python
# Ensure certain critical tools are always available
middleware = LLMToolSelectorMiddleware(
    max_tools=3,
    always_include=["error_handler", "logger"],
    system_prompt="Select the most relevant tools. The error_handler and logger are already included."
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[*critical_tools, *optional_tools],
    middleware=[middleware],
)

# error_handler and logger will always be present,
# plus 3 additional tools selected by the LLM
```

### Example 4: Custom System Prompt

```python
middleware = LLMToolSelectorMiddleware(
    max_tools=5,
    system_prompt="""
    You are an expert tool selector. Analyze the user's query and select
    the most appropriate tools. Prioritize:
    1. Tools that directly answer the question
    2. Supporting tools that provide context
    3. Tools that can verify information

    Return tools in order of relevance.
    """
)
```

## Related Pages

- `AgentMiddleware` - Base class for agent middleware
- `ModelRequest` - Request object containing messages and tools
- `ModelResponse` - Response object from model invocation
- `BaseTool` - Base class for LangChain tools
- `init_chat_model` - Function to initialize chat models from strings
