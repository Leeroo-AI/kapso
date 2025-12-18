{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|OpenAI Structured Outputs|https://platform.openai.com/docs/guides/structured-outputs]]
* [[source::Doc|Strategy Pattern|https://refactoring.guru/design-patterns/strategy]]
|-
! Domains
| [[domain::LLM]], [[domain::Structured_Output]], [[domain::Design_Patterns]], [[domain::Type_Safety]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Process of choosing the appropriate extraction mechanism (tool-based, provider-native, or automatic) for converting LLM responses into structured data.

=== Description ===

Strategy Selection determines how structured output will be extracted from language model responses. The choice affects:
* **Reliability:** Tool-based strategies have built-in error handling and retry
* **Performance:** Provider-native strategies may be faster with supporting models
* **Compatibility:** Not all providers support native JSON mode
* **Flexibility:** Tool strategies work with Union types and multiple schemas

The three available strategies form a union type that agents accept for their `response_format` parameter.

=== Usage ===

Select strategies based on:
* **ToolStrategy:** Default choice; works everywhere; supports `handle_errors` for retry
* **ProviderStrategy:** When using OpenAI/compatible providers; want strict JSON enforcement
* **AutoStrategy:** Cross-provider agents; let LangChain optimize strategy selection

Decision factors:
* Provider capabilities (does it support JSON mode?)
* Error handling requirements (need retry on validation errors?)
* Schema complexity (Union types require ToolStrategy)
* Performance requirements (native mode may be faster)

== Theoretical Basis ==

Strategy Selection implements **Strategy Pattern** for polymorphic structured output extraction.

'''1. Strategy Union Type'''

<syntaxhighlight lang="python">
from langchain.agents.structured_output import (
    ToolStrategy,
    ProviderStrategy,
    AutoStrategy,
)

# ResponseFormat is a union of all strategies
ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]

# Used in agent configuration
def create_agent(
    model: ...,
    tools: ...,
    response_format: ResponseFormat | None = None,
) -> ...:
    """response_format accepts any strategy type."""
</syntaxhighlight>

'''2. Selection Decision Tree'''

<syntaxhighlight lang="text">
                    ┌─────────────────────┐
                    │ Need Structured     │
                    │ Output?             │
                    └─────────┬───────────┘
                              │ Yes
                    ┌─────────▼───────────┐
                    │ Known Provider?     │
                    └─────────┬───────────┘
                     No │           │ Yes
         ┌──────────────┘           └──────────────┐
         │                                         │
┌────────▼────────┐                   ┌────────────▼────────────┐
│ AutoStrategy    │                   │ Provider Supports       │
│ (auto-detect)   │                   │ JSON Mode?              │
└─────────────────┘                   └────────────┬────────────┘
                                       No │           │ Yes
                          ┌───────────────┘           └───────────────┐
                          │                                           │
                 ┌────────▼────────┐                     ┌────────────▼────────────┐
                 │ ToolStrategy    │                     │ Need Error Retry?       │
                 │ (fallback)      │                     └────────────┬────────────┘
                 └─────────────────┘                      Yes │           │ No
                                              ┌───────────────┘           └──────────┐
                                              │                                      │
                                     ┌────────▼────────┐                ┌────────────▼──────────┐
                                     │ ToolStrategy    │                │ ProviderStrategy      │
                                     │ (handle_errors) │                │ (strict mode)         │
                                     └─────────────────┘                └───────────────────────┘
</syntaxhighlight>

'''3. Strategy Capabilities Comparison'''

<syntaxhighlight lang="python">
# ToolStrategy capabilities
tool_strategy = ToolStrategy(
    Schema,
    handle_errors=True,  # Retry on validation errors
    tool_message_content="Structured output received",  # Custom message
)
# - Works with any provider
# - Supports Union types (schema variants)
# - Built-in error handling and retry
# - Model "calls" a synthetic tool

# ProviderStrategy capabilities
provider_strategy = ProviderStrategy(
    Schema,
    strict=True,  # Strict schema enforcement
)
# - Uses OpenAI JSON mode
# - Fastest for supported providers
# - Strictest validation
# - Limited to single schema (no unions)

# AutoStrategy capabilities
auto_strategy = AutoStrategy(Schema)
# - Auto-selects based on model
# - Best for configurable models
# - Falls back gracefully
# - No manual tuning needed
</syntaxhighlight>

'''4. Strategy Selection by Use Case'''

<syntaxhighlight lang="python">
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy, AutoStrategy
from pydantic import BaseModel

class Output(BaseModel):
    result: str

# Use Case 1: Multi-provider agent
# AutoStrategy auto-selects per invocation
response_format = AutoStrategy(Output)

# Use Case 2: OpenAI-only with strict validation
# ProviderStrategy for native JSON mode
response_format = ProviderStrategy(Output, strict=True)

# Use Case 3: Need error retry
# ToolStrategy with handle_errors
response_format = ToolStrategy(Output, handle_errors=True)

# Use Case 4: Union types (multiple possible outputs)
from typing import Union

class Success(BaseModel):
    data: dict

class Error(BaseModel):
    message: str

# Only ToolStrategy supports Union
response_format = ToolStrategy(Union[Success, Error])
</syntaxhighlight>

'''5. Runtime Strategy Resolution (AutoStrategy)'''

<syntaxhighlight lang="python">
# Pseudo-code for AutoStrategy resolution
def resolve_auto_strategy(
    auto: AutoStrategy,
    model: BaseChatModel
) -> ToolStrategy | ProviderStrategy:
    """AutoStrategy resolves at runtime based on model."""
    # Check if model supports native structured output
    if hasattr(model, "supports_structured_output"):
        if model.supports_structured_output:
            return ProviderStrategy(auto.schema)

    # Check model provider
    model_name = getattr(model, "model_name", "")
    if "gpt" in model_name.lower() or "openai" in model_name.lower():
        return ProviderStrategy(auto.schema)

    # Fallback to tool strategy
    return ToolStrategy(auto.schema)
</syntaxhighlight>

'''6. Strategy in Agent Configuration'''

<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# Strategy passed to create_agent
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[search_tool, calculator_tool],
    response_format=ToolStrategy(ResponseSchema, handle_errors=True),
)

# Agent uses strategy to:
# 1. Configure model binding (bind_tools or response_format)
# 2. Create output parsing binding
# 3. Handle errors during parsing
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_ResponseFormat_type_union]]

=== Used By Workflows ===
* Structured_Output_Workflow (Step 2)
