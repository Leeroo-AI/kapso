{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|Python Type Unions|https://docs.python.org/3/library/typing.html#typing.Union]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Structured_Output]], [[domain::Type_Safety]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for selecting structured output extraction strategies through the ResponseFormat type union, provided by LangChain's structured output system.

=== Description ===

`ResponseFormat` is a type alias representing the union of all available structured output strategies: `ToolStrategy`, `ProviderStrategy`, and `AutoStrategy`. This union enables type-safe strategy selection where:

* Each strategy variant has distinct initialization parameters
* Agents accept any strategy through a single `response_format` parameter
* Type checkers validate strategy-specific options

The type union pattern allows polymorphic handling of different extraction mechanisms.

=== Usage ===

Use `ResponseFormat` (the type union) when:
* Defining type hints for strategy parameters
* Building generic functions that accept any strategy
* Implementing strategy pattern for structured output

Strategy selection guidelines:
* **ToolStrategy:** Works everywhere, supports Union schemas, has error retry
* **ProviderStrategy:** Fastest with OpenAI/compatible, strictest validation
* **AutoStrategy:** Best for configurable models, auto-selects per invocation

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/structured_output.py
* '''Lines:''' L443 (type alias), L181-243 (ToolStrategy), L246-286 (ProviderStrategy), L429-441 (AutoStrategy)

=== Signature ===
<syntaxhighlight lang="python">
# Type alias for all strategy variants
ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]


@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    """Use tool calling for structured output."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        tool_message_content: str | None = None,
        handle_errors: bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str] = True,
    ) -> None: ...


@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    """Use provider's native structured output."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        strict: bool | None = None,
    ) -> None: ...


class AutoStrategy(Generic[SchemaT]):
    """Automatically select best strategy."""

    def __init__(self, schema: type[SchemaT]) -> None: ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import (
    ToolStrategy,
    ProviderStrategy,
    AutoStrategy,
)

# ResponseFormat is typically used in type hints
def configure_agent(response_format: ToolStrategy | ProviderStrategy | AutoStrategy) -> None:
    ...
</syntaxhighlight>

== I/O Contract ==

=== Inputs (ToolStrategy) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema || type[SchemaT] || Yes || Output schema (Pydantic, dataclass, TypedDict, JSON dict)
|-
| tool_message_content || str | None || No || Content for ToolMessage (default: None)
|-
| handle_errors || bool | str | type | tuple | Callable || No || Error handling config (default: True)
|}

=== Inputs (ProviderStrategy) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema || type[SchemaT] || Yes || Output schema (Pydantic, dataclass, TypedDict, JSON dict)
|-
| strict || bool | None || No || Enable strict provider validation (default: None)
|}

=== Inputs (AutoStrategy) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema || type[SchemaT] || Yes || Output schema (Pydantic, dataclass, TypedDict, JSON dict)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| instance || ResponseFormat[SchemaT] || Configured strategy ready for agent use
|}

== Usage Examples ==

=== Selecting ToolStrategy ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Search results to return."""
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    snippet: str = Field(description="Result snippet")


# Select ToolStrategy for universal compatibility
response_format = ToolStrategy(
    SearchResult,
    handle_errors=True,  # Retry on validation errors
)

# Can also customize error handling
response_format = ToolStrategy(
    SearchResult,
    handle_errors="Please format output as valid SearchResult JSON.",
)
</syntaxhighlight>

=== Selecting ProviderStrategy ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel


class ExtractedEntity(BaseModel):
    """Entity extracted from text."""
    name: str
    type: str
    confidence: float


# Select ProviderStrategy for OpenAI with strict mode
response_format = ProviderStrategy(
    ExtractedEntity,
    strict=True,  # Enable strict JSON schema enforcement
)

# Without strict mode
response_format = ProviderStrategy(ExtractedEntity)
</syntaxhighlight>

=== Selecting AutoStrategy ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import AutoStrategy
from pydantic import BaseModel


class Classification(BaseModel):
    """Text classification."""
    label: str
    score: float


# AutoStrategy for configurable/multi-provider agents
response_format = AutoStrategy(Classification)

# Best when model is selected at runtime
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model(configurable_fields=("model",)),
    tools=[],
    response_format=AutoStrategy(Classification),
)

# Works with any provider
result1 = agent.invoke(msg, config={"configurable": {"model": "gpt-4o"}})
result2 = agent.invoke(msg, config={"configurable": {"model": "claude-sonnet-4-5-20250929"}})
</syntaxhighlight>

=== Union Schema with ToolStrategy ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel
from typing import Union, Literal


class SuccessResponse(BaseModel):
    """Successful result."""
    status: Literal["success"]
    data: dict


class ErrorResponse(BaseModel):
    """Error result."""
    status: Literal["error"]
    message: str


# Only ToolStrategy supports Union types
response_format = ToolStrategy(
    Union[SuccessResponse, ErrorResponse],
    handle_errors=True,
)

# Creates schema_specs for both variants
# Model can call either variant's tool
</syntaxhighlight>

=== Strategy Selection Function ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import (
    ToolStrategy,
    ProviderStrategy,
    AutoStrategy,
)
from pydantic import BaseModel


def select_strategy(
    schema: type[BaseModel],
    *,
    provider: str | None = None,
    need_retry: bool = True,
    is_union: bool = False,
) -> ToolStrategy | ProviderStrategy | AutoStrategy:
    """Select appropriate strategy based on requirements."""
    # Union types require ToolStrategy
    if is_union:
        return ToolStrategy(schema, handle_errors=need_retry)

    # Unknown provider: use AutoStrategy
    if provider is None:
        return AutoStrategy(schema)

    # OpenAI-compatible: prefer ProviderStrategy if no retry needed
    if provider in ("openai", "azure") and not need_retry:
        return ProviderStrategy(schema, strict=True)

    # Default to ToolStrategy
    return ToolStrategy(schema, handle_errors=need_retry)


# Usage
strategy = select_strategy(MySchema, provider="openai", need_retry=False)
</syntaxhighlight>

=== Type-Safe Strategy Handling ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import (
    ToolStrategy,
    ProviderStrategy,
    AutoStrategy,
)


def process_strategy(
    strategy: ToolStrategy | ProviderStrategy | AutoStrategy
) -> dict:
    """Process strategy based on type."""
    if isinstance(strategy, ToolStrategy):
        return {
            "type": "tool",
            "schema_count": len(strategy.schema_specs),
            "handle_errors": strategy.handle_errors,
        }
    elif isinstance(strategy, ProviderStrategy):
        return {
            "type": "provider",
            "strict": strategy.schema_spec.strict,
            "model_kwargs": strategy.to_model_kwargs(),
        }
    elif isinstance(strategy, AutoStrategy):
        return {
            "type": "auto",
            "schema": strategy.schema,
        }
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Strategy_Selection]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
