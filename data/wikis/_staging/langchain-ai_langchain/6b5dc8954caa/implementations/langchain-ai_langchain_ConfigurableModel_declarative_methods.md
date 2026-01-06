{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Tool_Calling]], [[domain::Structured_Output]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for binding tools and structured output schemas to configurable models through queued declarative operations, provided by LangChain.

=== Description ===

`_ConfigurableModel.bind_tools` and `with_structured_output` are declarative methods that queue operations to be applied when the actual model is instantiated. Since configurable models don't have a concrete model until invocation, these methods store the operation and its arguments for later application.

This enables the full chat model API on configurable models despite deferred instantiation.

=== Usage ===

Use these methods when:
* Adding tools to a configurable model
* Setting up structured output on runtime-selectable models
* Building flexible agent pipelines
* Creating reusable model configurations

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/chat_models/base.py
* '''Lines:''' L569-604 (__getattr__ queuing mechanism)
* '''Lines:''' L931-944 (explicit method implementations if any)

=== Signature ===
<syntaxhighlight lang="python">
class _ConfigurableModel:
    def bind_tools(
        self,
        tools: Sequence[dict | type[BaseModel] | Callable | BaseTool],
        **kwargs: Any,
    ) -> "_ConfigurableModel":
        """Queue tool binding for application at model instantiation.

        Args:
            tools: Tools to bind (Pydantic models, functions, dicts, BaseTool)
            **kwargs: Additional binding kwargs (tool_choice, etc.)

        Returns:
            New ConfigurableModel with queued bind_tools operation
        """

    def with_structured_output(
        self,
        schema: dict | type[BaseModel],
        **kwargs: Any,
    ) -> "_ConfigurableModel":
        """Queue structured output configuration.

        Args:
            schema: Output schema (Pydantic model or JSON schema dict)
            **kwargs: Additional structured output kwargs

        Returns:
            New ConfigurableModel with queued with_structured_output operation
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Methods available on configurable models
model = init_chat_model(configurable_fields=("model",))
model_with_tools = model.bind_tools([...])
model_with_output = model.with_structured_output(Schema)
</syntaxhighlight>

== I/O Contract ==

=== Inputs (bind_tools) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tools || Sequence[...] || Yes || Tools to bind (Pydantic, Callable, dict, BaseTool)
|-
| **kwargs || Any || No || tool_choice and other binding options
|}

=== Inputs (with_structured_output) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema || dict | type[BaseModel] || Yes || Output schema
|-
| **kwargs || Any || No || include_raw, method, strict, etc.
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || _ConfigurableModel || New configurable with queued operation
|}

== Usage Examples ==

=== Binding Tools to Configurable Model ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class SearchTool(BaseModel):
    """Search the web for information."""
    query: str = Field(description="Search query")


class WeatherTool(BaseModel):
    """Get current weather."""
    location: str = Field(description="City name")


# Create configurable model with tools
model = init_chat_model(configurable_fields=("model",))
model_with_tools = model.bind_tools([SearchTool, WeatherTool])

# Tools work with any configured model
response = model_with_tools.invoke(
    "What's the weather in NYC?",
    config={"configurable": {"model": "gpt-4o"}}
)
print(response.tool_calls)

# Same tools, different model
response = model_with_tools.invoke(
    "What's the weather in NYC?",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}}
)
</syntaxhighlight>

=== Structured Output on Configurable Model ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class Sentiment(BaseModel):
    """Sentiment analysis result."""
    label: str  # positive, negative, neutral
    confidence: float
    explanation: str


model = init_chat_model(configurable_fields=("model",))
structured_model = model.with_structured_output(Sentiment)

# Works with any provider
result = structured_model.invoke(
    "Analyze: 'This product is amazing!'",
    config={"configurable": {"model": "gpt-4o"}}
)
print(f"Sentiment: {result.label} ({result.confidence})")
</syntaxhighlight>

=== Chaining Declarative Operations ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class OutputSchema(BaseModel):
    answer: str
    sources: list[str]


# Chain multiple operations
model = (
    init_chat_model("gpt-4o", configurable_fields="any")
    .bind_tools([SearchTool, WeatherTool], tool_choice="auto")
    .with_config(config={"tags": ["production"]})
)

# All operations queued, applied at invocation
response = model.invoke("Search for Python tutorials")
</syntaxhighlight>

=== With Tool Choice ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

model = init_chat_model(configurable_fields=("model",))

# Force specific tool
model_force_search = model.bind_tools(
    [SearchTool, WeatherTool],
    tool_choice={"type": "function", "function": {"name": "SearchTool"}}
)

# Allow any tool or no tool
model_any = model.bind_tools(
    [SearchTool, WeatherTool],
    tool_choice="auto"
)

# Require tool use
model_required = model.bind_tools(
    [SearchTool, WeatherTool],
    tool_choice="required"
)
</syntaxhighlight>

=== Operation Queue Inspection ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

model = init_chat_model(configurable_fields=("model",))
model_with_ops = model.bind_tools([SearchTool]).with_structured_output(Schema)

# Inspect queued operations (internal attribute)
print(model_with_ops._queued_declarative_operations)
# [
#   ("bind_tools", ([SearchTool],), {}),
#   ("with_structured_output", (Schema,), {})
# ]
</syntaxhighlight>

=== Comparison with Fixed Model ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Fixed model: operations applied immediately
fixed = init_chat_model("gpt-4o")
fixed_with_tools = fixed.bind_tools([SearchTool])
# Type: ChatOpenAI (concrete model)

# Configurable model: operations queued
configurable = init_chat_model(configurable_fields=("model",))
configurable_with_tools = configurable.bind_tools([SearchTool])
# Type: _ConfigurableModel (wrapper with queued ops)

# Both work the same when invoked
response1 = fixed_with_tools.invoke("Search for X")
response2 = configurable_with_tools.invoke(
    "Search for X",
    config={"configurable": {"model": "gpt-4o"}}
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Declarative_Operation_Binding]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
