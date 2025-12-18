{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|OpenAI Structured Outputs|https://platform.openai.com/docs/guides/structured-outputs]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Structured_Output]], [[domain::Schema_Validation]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for configuring structured output extraction from language models using different strategy patterns, provided by LangChain.

=== Description ===

`ResponseFormat` strategies (`ToolStrategy`, `ProviderStrategy`, `AutoStrategy`) define how an agent extracts structured, typed responses from language models. Each strategy represents a different approach to forcing structured output:

* **ToolStrategy:** Creates a synthetic tool from the schema; model "calls" the tool with structured arguments
* **ProviderStrategy:** Uses provider-native JSON mode (OpenAI's `response_format`)
* **AutoStrategy:** Automatically selects the best strategy based on model capabilities

These strategies enable type-safe extraction of complex data structures from LLM responses.

=== Usage ===

Use `ResponseFormat` strategies when:
* Building agents that need typed output (not just text)
* Extracting structured data from unstructured text
* Enforcing output schemas for downstream processing
* Building reliable data pipelines with LLMs

Strategy selection guide:
* **ToolStrategy:** Best for most use cases; works across providers; supports error retry
* **ProviderStrategy:** Best when using OpenAI/compatible providers; strictest validation
* **AutoStrategy:** Best for cross-provider agents; lets LangChain choose

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/structured_output.py
* '''Lines:''' L181-443

=== Signature ===
<syntaxhighlight lang="python">
@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    """Use a tool calling strategy for model responses."""

    schema: type[SchemaT]
    """Schema for the tool calls."""

    schema_specs: list[_SchemaSpec[SchemaT]]
    """Schema specs for the tool calls."""

    tool_message_content: str | None
    """Content of ToolMessage returned when structured output tool is called."""

    handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
    """Error handling strategy. True=catch all, False=no retry, str=custom message."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        tool_message_content: str | None = None,
        handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str] = True,
    ) -> None:
        """Initialize ToolStrategy."""


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
        """Initialize ProviderStrategy.

        Args:
            schema: Schema to enforce via provider's native structured output.
            strict: Whether to request strict provider-side schema enforcement.
        """

    def to_model_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs to bind to a model."""


class AutoStrategy(Generic[SchemaT]):
    """Automatically select the best strategy for structured output."""

    schema: type[SchemaT]
    """Schema for automatic mode."""

    def __init__(self, schema: type[SchemaT]) -> None:
        """Initialize AutoStrategy with schema."""


# Type alias
ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy, AutoStrategy
</syntaxhighlight>

== I/O Contract ==

=== Inputs (ToolStrategy) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema || type[SchemaT] || Yes || Pydantic model, dataclass, TypedDict, or JSON schema dict
|-
| tool_message_content || str | None || No || Content to return in ToolMessage (default: None)
|-
| handle_errors || bool | str | ... || No || Error handling config (default: True for retry)
|}

=== Inputs (ProviderStrategy) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema || type[SchemaT] || Yes || Pydantic model, dataclass, TypedDict, or JSON schema dict
|-
| strict || bool | None || No || Whether to enforce strict validation on provider side
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| instance || ResponseFormat || Configured strategy ready for agent configuration
|}

== Usage Examples ==

=== ToolStrategy with Pydantic Model ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class WeatherResponse(BaseModel):
    """Structured weather response."""
    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions")
    humidity: int = Field(description="Humidity percentage")


agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    response_format=ToolStrategy(WeatherResponse),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in NYC?"}]
})

# result["structured_response"] is a WeatherResponse instance
weather: WeatherResponse = result["structured_response"]
print(f"Temperature: {weather.temperature}°C")
</syntaxhighlight>

=== ProviderStrategy for Strict JSON Mode ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class ExtractedData(BaseModel):
    """Data to extract."""
    name: str
    age: int
    email: str


# Use provider's native JSON schema enforcement
agent = create_agent(
    model=init_chat_model("openai:gpt-4o"),  # Works best with OpenAI
    tools=[],
    response_format=ProviderStrategy(ExtractedData, strict=True),
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract info: John Doe is 30 years old, email john@example.com"
    }]
})
</syntaxhighlight>

=== AutoStrategy for Cross-Provider Compatibility ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class Classification(BaseModel):
    """Text classification result."""
    category: str
    confidence: float


# Auto-selects best strategy based on model capabilities
agent = create_agent(
    model=init_chat_model(configurable_fields=("model",)),  # Configurable
    tools=[],
    response_format=AutoStrategy(Classification),
)

# Works with any provider
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Classify: 'Great product!'"}]},
    config={"configurable": {"model": "gpt-4o"}}  # Or "claude-sonnet-4-5-20250929", etc.
)
</syntaxhighlight>

=== ToolStrategy with Error Handling ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import ToolStrategy


# Catch all errors with custom message
strategy = ToolStrategy(
    MySchema,
    handle_errors="Please try again with valid JSON matching the schema."
)

# Only catch specific errors
from pydantic import ValidationError
strategy = ToolStrategy(
    MySchema,
    handle_errors=ValidationError  # Only retry on validation errors
)

# Custom error handler
strategy = ToolStrategy(
    MySchema,
    handle_errors=lambda e: f"Error: {e}. Please fix and retry."
)

# Disable retry (let errors propagate)
strategy = ToolStrategy(MySchema, handle_errors=False)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Structured_Output_Strategy]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
