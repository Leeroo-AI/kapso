{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|OpenAI Structured Outputs|https://platform.openai.com/docs/guides/structured-outputs]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Structured_Output]], [[domain::Tool_Calling]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for binding structured output configuration to models during agent execution, provided by LangChain's agent factory.

=== Description ===

`_get_bound_model` is an internal function in the agent factory that configures model binding based on the response format strategy. It handles:

* AutoStrategy resolution to concrete strategy based on model capabilities
* ToolStrategy binding with synthetic output tools and forced tool choice
* ProviderStrategy binding with native JSON schema response format
* Standard tool binding when no structured output is configured

This function is the execution point where strategy configuration becomes model configuration.

=== Usage ===

Use `_get_bound_model` (indirectly via agent execution) when:
* Agent's model node processes requests with structured output
* Configurable models need strategy auto-detection
* Combining user tools with structured output tools

The function is called automatically during agent invocation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/factory.py
* '''Lines:''' L976-1088

=== Signature ===
<syntaxhighlight lang="python">
def _get_bound_model(request: ModelRequest) -> tuple[Runnable, ResponseFormat | None]:
    """Get the model with appropriate tool bindings.

    Performs auto-detection of strategy if needed based on model capabilities.

    Args:
        request: The model request containing model, tools, and response format.

    Returns:
        Tuple of (bound_model, effective_response_format) where
        effective_response_format is the actual strategy used (may differ from
        initial if auto-detected).
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal function, not imported directly
# Used via agent invocation
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    response_format=strategy,  # Triggers _get_bound_model internally
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| request.model || BaseChatModel || Yes || The language model to bind
|-
| request.tools || list[BaseTool | dict] || Yes || User tools to bind
|-
| request.response_format || ResponseFormat | None || No || Structured output strategy
|-
| request.tool_choice || str | dict | None || No || Tool selection mode
|-
| request.model_settings || dict || No || Additional model kwargs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| bound_model || Runnable || Model with tools and response format bound
|-
| effective_response_format || ResponseFormat | None || Actual strategy used (may differ if auto-detected)
|}

== Usage Examples ==

=== ToolStrategy Model Binding ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Search results."""
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    relevance: float = Field(description="Relevance score 0-1")


# ToolStrategy binds synthetic tool with tool_choice="any"
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[search_tool],
    response_format=ToolStrategy(SearchResult, handle_errors=True),
)

# Internally, _get_bound_model:
# 1. Creates OutputToolBinding from SearchResult
# 2. Adds SearchResult tool to [search_tool]
# 3. Binds model with tool_choice="any"

result = agent.invoke({"messages": [{"role": "user", "content": "Search for Python tutorials"}]})
# result["structured_response"] is SearchResult instance
</syntaxhighlight>

=== ProviderStrategy Model Binding ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class ExtractedData(BaseModel):
    """Extracted information."""
    name: str
    age: int
    occupation: str


# ProviderStrategy uses native JSON mode
agent = create_agent(
    model=init_chat_model("openai:gpt-4o"),
    tools=[],
    response_format=ProviderStrategy(ExtractedData, strict=True),
)

# Internally, _get_bound_model:
# 1. Calls provider_strategy.to_model_kwargs()
# 2. Gets {"response_format": {"type": "json_schema", ...}}
# 3. Binds model with strict=True and response_format

result = agent.invoke({"messages": [
    {"role": "user", "content": "Extract: John is 30, works as engineer"}
]})
</syntaxhighlight>

=== AutoStrategy Resolution ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class Output(BaseModel):
    answer: str


# AutoStrategy auto-detects best approach
agent = create_agent(
    model=init_chat_model(configurable_fields=("model",)),
    tools=[],
    response_format=AutoStrategy(Output),
)

# Internally, _get_bound_model:
# 1. Detects AutoStrategy
# 2. Checks if model supports provider strategy
# 3. Converts to ProviderStrategy or ToolStrategy

# With OpenAI model (supports JSON mode)
result1 = agent.invoke(
    {"messages": [msg]},
    config={"configurable": {"model": "gpt-4o"}}
)
# Uses ProviderStrategy

# With model without JSON mode support
result2 = agent.invoke(
    {"messages": [msg]},
    config={"configurable": {"model": "other-model"}}
)
# Falls back to ToolStrategy
</syntaxhighlight>

=== Combined User Tools and Structured Output ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel


@tool
def calculator(expression: str) -> float:
    """Calculate math expression."""
    return eval(expression)


class MathResult(BaseModel):
    """Math calculation result."""
    expression: str
    result: float
    explanation: str


# User tools combined with structured output tool
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[calculator],  # User tool
    response_format=ToolStrategy(MathResult),  # Adds MathResult tool
)

# _get_bound_model binds [calculator, MathResult] with tool_choice="any"
# Model can use calculator, then must call MathResult for final answer

result = agent.invoke({"messages": [
    {"role": "user", "content": "Calculate 15 * 7 and explain"}
]})
</syntaxhighlight>

=== Model Binding Without Structured Output ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


# No response_format - standard tool binding
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[search_tool, calculator_tool],
    response_format=None,  # No structured output
)

# _get_bound_model:
# 1. Binds model with user tools only
# 2. Uses user's tool_choice (default or specified)
# 3. Returns (bound_model, None)

result = agent.invoke({"messages": [msg]})
# result["messages"][-1] is AIMessage (not structured)
</syntaxhighlight>

=== Effective Response Format Return ===
<syntaxhighlight lang="python">
# _get_bound_model returns the effective strategy
# This is important for parsing responses correctly

# Example: AutoStrategy becomes ProviderStrategy
request = ModelRequest(
    model=openai_model,
    tools=[],
    response_format=AutoStrategy(Schema),
)

bound_model, effective_format = _get_bound_model(request)
# effective_format is ProviderStrategy (not AutoStrategy)

# Use effective_format for response parsing
if isinstance(effective_format, ProviderStrategy):
    # Parse from message content
    result = provider_binding.parse(response)
elif isinstance(effective_format, ToolStrategy):
    # Parse from tool call args
    result = tool_binding.parse(tool_call.args)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Model_Invocation_With_Schema]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
