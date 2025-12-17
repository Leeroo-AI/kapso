# BaseTool Creation

**Sources:**
- `langchain_core.tools` module (external - typically at `../langchain-core/` relative to monorepo)
- Integration test examples across `libs/partners/*/tests/`
- LangChain Core API Reference

**Domains:** Agent Tools, Function Calling, Schema Generation

**Last Updated:** 2025-12-17

---

## Overview

BaseTool creation encompasses the mechanisms for converting Python functions into LLM-invocable tools through the `@tool` decorator, `BaseTool` subclassing, or `StructuredTool` instantiation. These tools provide the functional capabilities that LLM agents use to interact with external systems and perform computations.

## Description

The BaseTool creation implementation provides three primary approaches for defining tools:

1. **Decorator-based (`@tool`)** - The most concise approach for simple functions
2. **Class-based (`BaseTool` subclass)** - For complex tools requiring custom logic, state, or async support
3. **Factory-based (`StructuredTool`)** - For programmatic tool creation from schemas

All approaches generate:
- A JSON schema describing parameters
- Validation logic for arguments
- Invocation wrappers for sync/async execution
- Integration with LangChain's message-based conversation model

The implementation handles:
- Automatic schema generation from type hints and Pydantic models
- Docstring parsing for tool and parameter descriptions
- Argument validation before execution
- Conversion of return values to `ToolMessage` objects
- Error handling and exception propagation

### Key Design Features

**Unified Interface**
All tool creation methods produce instances conforming to the `BaseTool` protocol, ensuring consistent behavior across different definition styles.

**Schema-First Design**
Tools are defined by their input schema (typically JSON Schema), which serves as the contract between the LLM and the implementation.

**Sync/Async Support**
Tools can implement both synchronous (`_run`) and asynchronous (`_arun`) execution methods, with the framework handling appropriate invocation.

**Optional Features**
Tools support optional configuration:
- `return_direct`: Return tool result directly to user without further LLM processing
- `handle_tool_error`: Error handling strategy (propagate, catch, custom message)
- `response_format`: Expected return type specification
- `extras`: Provider-specific metadata (caching, deferral, etc.)

## Code Reference

### Location
**Module:** `langchain_core.tools` (external dependency)
**Typical Import:** `from langchain_core.tools import tool, BaseTool, StructuredTool`

### Core Types

```python
# From langchain_core.tools

class BaseTool(ABC):
    """Base class for tools."""

    name: str
    """The unique name of the tool."""

    description: str
    """Used to tell the model how/when/why to use the tool."""

    args_schema: type[BaseModel] | None = None
    """The schema for the tool's arguments."""

    return_direct: bool = False
    """Whether to return the tool's output directly."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool synchronously."""
        raise NotImplementedError

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool asynchronously."""
        raise NotImplementedError
```

## I/O Contract

### Inputs (Tool Creation)

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` (decorator) | `Callable` | Python function to convert to a tool |
| `name` | `str` | Tool name for LLM (defaults to function name) |
| `description` | `str` | What the tool does (defaults to docstring) |
| `args_schema` | `type[BaseModel]` | Pydantic model for argument validation |
| `return_direct` | `bool` | Skip LLM processing of tool result |
| `handle_tool_error` | `bool \| str \| Callable` | Error handling strategy |

### Outputs (Tool Creation)

| Type | Description |
|------|-------------|
| `BaseTool` | Tool instance ready for binding to chat models |

### Runtime I/O (Tool Invocation)

**Input:** Tool call from LLM with `name`, `args` dict, and `tool_call_id`
**Output:** `ToolMessage` with execution result and same `tool_call_id`

## Usage Examples

### Example 1: Simple Function Tool with @tool Decorator

```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g. 'San Francisco, CA'

    Returns:
        Weather description string
    """
    # In reality, would call a weather API
    return f"Sunny and 72°F in {location}"

# Tool is ready to use
print(get_weather.name)  # "get_weather"
print(get_weather.description)  # Extracted from docstring
print(get_weather.args_schema)  # Auto-generated from type hints
```

### Example 2: Tool with Pydantic Schema

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    location: str = Field(..., description="City and state, e.g. 'San Francisco, CA'")
    unit: str = Field("fahrenheit", description="Temperature unit: 'celsius' or 'fahrenheit'")

@tool(args_schema=WeatherInput)
def get_weather_detailed(location: str, unit: str = "fahrenheit") -> str:
    """Get detailed weather information for a location."""
    temp = "22°C" if unit == "celsius" else "72°F"
    return f"Sunny and {temp} in {location}"

# More precise schema for LLM
print(get_weather_detailed.args_schema.model_json_schema())
```

### Example 3: Class-Based Tool with Custom Logic

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    operation: str = Field(..., description="Math operation: add, subtract, multiply, divide")
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

class CalculatorTool(BaseTool):
    """Tool for performing arithmetic operations."""

    name: str = "calculator"
    description: str = "Perform basic arithmetic operations on two numbers"
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, operation: str, a: float, b: float) -> str:
        """Execute the calculation."""
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: Division by zero"
        }
        result = operations.get(operation, "Error: Unknown operation")
        return f"Result: {result}"

    async def _arun(self, operation: str, a: float, b: float) -> str:
        """Async version (calls sync version for CPU-bound operations)."""
        return self._run(operation, a, b)

calculator = CalculatorTool()
```

### Example 4: Tool with Error Handling

```python
from langchain_core.tools import tool
import requests

@tool(
    handle_tool_error=True,  # Return error message instead of raising
    return_direct=False  # Let LLM process the result
)
def fetch_webpage(url: str) -> str:
    """Fetch content from a webpage.

    Args:
        url: The URL to fetch

    Returns:
        Page content or error message
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text[:1000]  # First 1000 chars
    except requests.RequestException as e:
        # With handle_tool_error=True, this becomes a ToolMessage with error
        raise ValueError(f"Failed to fetch {url}: {str(e)}")
```

### Example 5: Async Tool for I/O Operations

```python
from langchain_core.tools import tool
import aiohttp

@tool
async def async_fetch_data(api_endpoint: str) -> str:
    """Asynchronously fetch data from an API endpoint.

    Args:
        api_endpoint: Full URL of the API endpoint

    Returns:
        JSON response as string
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(api_endpoint) as response:
            data = await response.json()
            return str(data)

# This tool only has async implementation
# Calling .invoke() will raise NotImplementedError
# Must use .ainvoke() instead
```

### Example 6: Using Tools in Agent

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the product database.

    Args:
        query: Search query string
    """
    # Simulate database search
    return f"Found 3 products matching '{query}'"

@tool
def process_order(product_id: str, quantity: int) -> str:
    """Process a customer order.

    Args:
        product_id: Product identifier
        quantity: Number of items to order
    """
    return f"Order placed: {quantity}x {product_id}"

# Create agent with tools
agent = create_agent(
    model="gpt-4o",
    tools=[search_database, process_order],
    system_prompt="You are a helpful shopping assistant."
)

# Agent can now use these tools
result = agent.invoke({
    "messages": [{"role": "user", "content": "Find laptops and order 2 of the best one"}]
})
```

### Example 7: Provider-Specific Tool Features

```python
from langchain_core.tools import tool

@tool(
    extras={
        "cache_control": {"type": "ephemeral"},  # Anthropic caching
        "defer_loading": True  # Anthropic deferred tool loading
    }
)
def expensive_computation(data: str) -> str:
    """Perform expensive computation that benefits from caching.

    Args:
        data: Input data to process
    """
    # Complex computation here
    return f"Processed: {data}"

# The extras dict is passed to provider-specific tool binding logic
```

## Related Pages

**Principle:**
- [[langchain-ai_langchain_Tool_Definition]] - Core principle of tool definition

**Related Implementations:**
- [[langchain-ai_langchain_ToolNode]] - Executes tool calls in agent graphs
- [[langchain-ai_langchain_wrap_tool_call]] - Middleware for intercepting tool execution
- [[langchain-ai_langchain_bind_tools]] - Binding tools to chat models

**Used In:**
- [[langchain-ai_langchain_create_agent]] - Tools are passed to agent creation
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Tool definition is Step 2

**Workflows:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Uses tools in agent workflows
- [[langchain-ai_langchain_Middleware_Composition]] - Middleware can register tools
