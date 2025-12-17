# ResponseFormat Strategies

**Sources:**
- `libs/langchain_v1/langchain/agents/structured_output.py:L1-443`
- Usage in `libs/langchain_v1/langchain/agents/factory.py`

**Domains:** Structured Output, Schema Validation, LLM Constraints

**Last Updated:** 2025-12-17

---

## Overview

ResponseFormat strategies (`ToolStrategy`, `ProviderStrategy`, `AutoStrategy`) are configuration objects that control how LLM agents produce structured outputs conforming to predefined schemas. They encapsulate the mechanism for constraining and validating LLM responses.

## Description

The ResponseFormat implementation provides three strategy classes:

1. **`ToolStrategy`** - Implements structured output by binding the schema as an artificial tool that the LLM must call. Works with any model supporting tool calling. Supports Union types (multiple possible response structures).

2. **`ProviderStrategy`** - Uses provider-native structured output APIs (e.g., OpenAI's JSON Schema mode). Provides stronger guarantees but requires provider support.

3. **`AutoStrategy`** - Automatically selects between Tool and Provider strategies based on model capabilities. Simplifies the API while optimizing for best results.

All strategies work with:
- Pydantic `BaseModel` classes
- Python `dataclass` decorated classes
- `TypedDict` types
- Raw JSON Schema dictionaries

The implementation handles:
- Schema conversion to tool definitions (ToolStrategy)
- Schema conversion to provider-specific formats (ProviderStrategy)
- Response parsing and validation
- Error handling and retry logic
- Union type decomposition

## Code Reference

### Location
**File:** `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/structured_output.py`
**Lines:** 1-443

### Core Classes

```python
@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    """Use a tool calling strategy for model responses."""

    schema: type[SchemaT]
    """Schema for the tool calls."""

    schema_specs: list[_SchemaSpec[SchemaT]]
    """Schema specs for the tool calls (handles Union decomposition)."""

    tool_message_content: str | None
    """Content of tool message returned when model calls structured output tool."""

    handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
    """Error handling strategy for validation failures."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        tool_message_content: str | None = None,
        handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str] = True,
    ) -> None:
        """Initialize ToolStrategy with schema and error handling."""
```

```python
@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    """Use the model provider's native structured output method."""

    schema: type[SchemaT]
    """Schema for native mode."""

    schema_spec: _SchemaSpec[SchemaT]
    """Schema spec for native mode."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        strict: bool | None = None,
    ) -> None:
        """Initialize ProviderStrategy with schema.

        Args:
            schema: Schema to enforce via the provider's native structured output.
            strict: Whether to request strict provider-side schema enforcement.
        """

    def to_model_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs to bind to a model to force structured output."""
        # Returns OpenAI-style response_format configuration
```

```python
class AutoStrategy(Generic[SchemaT]):
    """Automatically select the best strategy for structured output."""

    schema: type[SchemaT]
    """Schema for automatic mode."""

    def __init__(self, schema: type[SchemaT]) -> None:
        """Initialize AutoStrategy with schema."""
```

### Helper Classes

```python
@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""

    schema: type[SchemaT]
    name: str
    description: str
    schema_kind: SchemaKind  # "pydantic" | "dataclass" | "typeddict" | "json_schema"
    json_schema: dict[str, Any]
    strict: bool | None

@dataclass
class OutputToolBinding(Generic[SchemaT]):
    """Information for tracking structured output tool metadata."""

    schema: type[SchemaT]
    schema_kind: SchemaKind
    tool: BaseTool

    def parse(self, tool_args: dict[str, Any]) -> SchemaT:
        """Parse tool arguments according to the schema."""

@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    """Information for tracking native structured output metadata."""

    schema: type[SchemaT]
    schema_kind: SchemaKind

    def parse(self, response: AIMessage) -> SchemaT:
        """Parse AIMessage content according to the schema."""
```

### Error Types

```python
class StructuredOutputError(Exception):
    """Base class for structured output errors."""
    ai_message: AIMessage

class MultipleStructuredOutputsError(StructuredOutputError):
    """Raised when model returns multiple structured output tool calls when only one is expected."""

class StructuredOutputValidationError(StructuredOutputError):
    """Raised when structured output tool call arguments fail to parse according to the schema."""
```

## I/O Contract

### Strategy Creation

**ToolStrategy:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | `type[BaseModel] \| dataclass \| TypedDict \| dict` | Response schema |
| `tool_message_content` | `str \| None` | Custom tool message content |
| `handle_errors` | `bool \| str \| Callable` | Error handling strategy |

**ProviderStrategy:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | `type[BaseModel] \| dataclass \| TypedDict \| dict` | Response schema |
| `strict` | `bool \| None` | Enable strict schema enforcement |

**AutoStrategy:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | `type[BaseModel] \| dataclass \| TypedDict \| dict` | Response schema |

### Runtime Behavior

**Input:** LLM response (either tool call or native JSON)
**Output:** Parsed instance of `schema` type, stored in `state["structured_response"]`
**Exceptions:** `StructuredOutputValidationError`, `MultipleStructuredOutputsError`

## Usage Examples

### Example 1: Basic Structured Output with AutoStrategy

```python
from langchain.agents import create_agent
from pydantic import BaseModel, Field

class WeatherResponse(BaseModel):
    """Structured weather information."""
    location: str = Field(..., description="City name")
    temperature: float = Field(..., description="Temperature in Fahrenheit")
    conditions: str = Field(..., description="Weather conditions (e.g., 'sunny', 'rainy')")

# AutoStrategy automatically selects best approach
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=WeatherResponse  # Implicitly wrapped in AutoStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in NYC?"}]
})

# Access structured response
weather: WeatherResponse = result["structured_response"]
print(f"Temperature: {weather.temperature}°F")
print(f"Conditions: {weather.conditions}")
```

### Example 2: Explicit ToolStrategy with Error Handling

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

class DatabaseQuery(BaseModel):
    """A database query."""
    table: str = Field(..., description="Table name to query")
    columns: list[str] = Field(..., description="Columns to select")
    where_clause: str = Field(None, description="Optional WHERE clause")

# Use ToolStrategy with custom error handling
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=ToolStrategy(
        schema=DatabaseQuery,
        handle_errors=True,  # Retry on validation errors
        tool_message_content="Query structured successfully"
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Get all users from the users table"}]
})

query: DatabaseQuery = result["structured_response"]
print(f"SELECT {', '.join(query.columns)} FROM {query.table}")
```

### Example 3: ProviderStrategy for Maximum Reliability

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel, Field

class APIRequest(BaseModel):
    """Structured API request."""
    endpoint: str = Field(..., description="API endpoint path")
    method: str = Field(..., description="HTTP method: GET, POST, PUT, DELETE")
    body: dict = Field(default_factory=dict, description="Request body")

# Use ProviderStrategy for strict validation
agent = create_agent(
    model="gpt-4o",  # OpenAI supports provider strategy
    tools=[...],
    response_format=ProviderStrategy(
        schema=APIRequest,
        strict=True  # Enable strict mode
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Create a POST request to /api/users"}]
})

request: APIRequest = result["structured_response"]
```

### Example 4: Union Types with ToolStrategy

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from typing import Union

class SuccessResponse(BaseModel):
    """Successful operation result."""
    status: str = Field("success", description="Always 'success'")
    data: dict = Field(..., description="Result data")

class ErrorResponse(BaseModel):
    """Error result."""
    status: str = Field("error", description="Always 'error'")
    error_message: str = Field(..., description="Error description")

# Union type automatically decomposed into multiple tools
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=ToolStrategy(
        schema=Union[SuccessResponse, ErrorResponse]
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Process this request"}]
})

# Response is either SuccessResponse or ErrorResponse
response = result["structured_response"]
if isinstance(response, SuccessResponse):
    print(f"Success: {response.data}")
else:
    print(f"Error: {response.error_message}")
```

### Example 5: Dataclass Schema

```python
from langchain.agents import create_agent
from dataclasses import dataclass
from typing import List

@dataclass
class TodoItem:
    """A todo item."""
    title: str
    description: str
    priority: int  # 1-5
    tags: List[str]

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=TodoItem  # Dataclass works directly
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Create a todo for buying groceries"}]
})

todo: TodoItem = result["structured_response"]
print(f"Created: {todo.title} (Priority: {todo.priority})")
```

### Example 6: TypedDict Schema

```python
from langchain.agents import create_agent
from typing import TypedDict, List

class UserProfile(TypedDict):
    """User profile information."""
    username: str
    email: str
    roles: List[str]
    active: bool

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=UserProfile  # TypedDict works directly
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract user info from conversation"}]
})

profile: UserProfile = result["structured_response"]
print(f"User: {profile['username']}")
```

### Example 7: Custom Error Handler

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

class ProductInfo(BaseModel):
    """Product information."""
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Price in USD")
    in_stock: bool = Field(..., description="Whether product is in stock")

def custom_error_handler(error: Exception) -> str:
    """Custom error message for LLM."""
    return f"The response format was invalid. Please ensure all fields are provided. Error: {str(error)}"

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=ToolStrategy(
        schema=ProductInfo,
        handle_errors=custom_error_handler  # Custom error handling
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Tell me about the laptop"}]
})

product: ProductInfo = result["structured_response"]
```

### Example 8: No Error Handling (Fail Fast)

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, StructuredOutputValidationError
from pydantic import BaseModel

class CriticalData(BaseModel):
    """Data that must be valid."""
    value: int

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=ToolStrategy(
        schema=CriticalData,
        handle_errors=False  # Raise exception on validation error
    )
)

try:
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Generate data"}]
    })
    data: CriticalData = result["structured_response"]
except StructuredOutputValidationError as e:
    print(f"Validation failed: {e}")
    print(f"AI message: {e.ai_message.content}")
```

## Related Pages

**Principle:**
- [[langchain-ai_langchain_Structured_Output_Configuration]] - Core principle implemented

**Related Implementations:**
- [[langchain-ai_langchain_SchemaSpec]] - Internal schema representation
- [[langchain-ai_langchain_OutputToolBinding]] - Tool binding for schemas
- [[langchain-ai_langchain_ProviderStrategyBinding]] - Provider binding for schemas
- [[langchain-ai_langchain_parse_with_schema]] - Schema validation logic

**Used In:**
- [[langchain-ai_langchain_create_agent]] - Accepts `response_format` parameter
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Step 4 of workflow

**Workflows:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Structured output is Step 4

**Related Principles:**
- [[langchain-ai_langchain_Tool_Definition]] - ToolStrategy builds on tool calling
