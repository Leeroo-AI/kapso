{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|OpenAI Structured Outputs|https://platform.openai.com/docs/guides/structured-outputs]]
* [[source::Doc|LangChain Tool Binding|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Structured_Output]], [[domain::Tool_Calling]], [[domain::API_Integration]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Process of binding structured output configuration to a model and invoking it to produce schema-compliant responses.

=== Description ===

Model Invocation With Schema is the execution step where the configured strategy is applied to the model. The invocation differs based on strategy:

* **ToolStrategy:** Model is bound with synthetic output tools, `tool_choice="any"` forces tool use
* **ProviderStrategy:** Model is bound with `response_format` containing JSON schema
* **AutoStrategy:** Strategy is auto-detected at runtime, then invoked as above

The bound model produces responses that can be parsed back to typed instances.

=== Usage ===

Model invocation with schema happens when:
* Agent's model node processes a request with `response_format` set
* Direct model binding for structured extraction
* Configurable models resolving strategy at runtime

The invocation handles:
* Tool binding with user tools + structured output tools
* Response format configuration for provider-native mode
* Auto-detection of best strategy based on model capabilities

== Theoretical Basis ==

Model Invocation With Schema implements **Template Method** for strategy-specific model binding.

'''1. Strategy-Based Model Binding'''

<syntaxhighlight lang="python">
# ToolStrategy binding
if isinstance(response_format, ToolStrategy):
    # Add structured output tools to user tools
    final_tools = list(user_tools) + [binding.tool for binding in output_bindings]
    # Force tool use
    bound_model = model.bind_tools(final_tools, tool_choice="any")

# ProviderStrategy binding
if isinstance(response_format, ProviderStrategy):
    # Use native JSON mode
    kwargs = response_format.to_model_kwargs()
    # kwargs = {"response_format": {"type": "json_schema", "json_schema": {...}}}
    bound_model = model.bind_tools(user_tools, strict=True, **kwargs)

# No structured output
if response_format is None:
    bound_model = model.bind_tools(user_tools) if user_tools else model
</syntaxhighlight>

'''2. AutoStrategy Resolution'''

<syntaxhighlight lang="python">
# AutoStrategy auto-detects at invocation time
if isinstance(response_format, AutoStrategy):
    # Check if model supports provider strategy
    if _supports_provider_strategy(model, tools=tools):
        # Convert to ProviderStrategy
        effective_format = ProviderStrategy(schema=response_format.schema)
    else:
        # Fallback to ToolStrategy
        effective_format = ToolStrategy(schema=response_format.schema)
else:
    # Explicit strategy - preserve it
    effective_format = response_format
</syntaxhighlight>

'''3. _get_bound_model Function Flow'''

<syntaxhighlight lang="text">
ModelRequest                          Bound Model
┌─────────────────────┐              ┌─────────────────────┐
│ model: BaseChatModel│              │ model with:         │
│ tools: [Tool1, ...] │              │   - tools bound     │
│ response_format:    │   ────────►  │   - tool_choice set │
│   ToolStrategy      │              │   - or JSON mode    │
│ model_settings: {}  │              │                     │
└─────────────────────┘              └─────────────────────┘

Auto-Detection (if AutoStrategy):
┌───────────────────┐
│ Check model       │
│ capabilities      │───► ToolStrategy or ProviderStrategy
└───────────────────┘
</syntaxhighlight>

'''4. Tool Choice Configuration'''

<syntaxhighlight lang="python">
# ToolStrategy forces tool use to ensure structured output
tool_choice = "any"  # Model MUST call a tool

# This ensures the model produces structured output
# rather than free-form text

# Without structured output tools:
tool_choice = request.tool_choice  # User's choice or default

# Combined: user tools + structured output tools
final_tools = user_tools + structured_output_tools
bound_model = model.bind_tools(final_tools, tool_choice="any")
</syntaxhighlight>

'''5. ProviderStrategy Model Kwargs'''

<syntaxhighlight lang="python">
# ProviderStrategy generates OpenAI-compatible kwargs
class ProviderStrategy:
    def to_model_kwargs(self) -> dict[str, Any]:
        json_schema = {
            "name": self.schema_spec.name,
            "schema": self.schema_spec.json_schema,
        }
        if self.schema_spec.strict:
            json_schema["strict"] = True

        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        }

# Applied to model
model.bind_tools(tools, **provider_strategy.to_model_kwargs())
</syntaxhighlight>

'''6. Invocation with Response Format'''

<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel


class Output(BaseModel):
    result: str
    confidence: float


# Agent created with response_format
agent = create_agent(
    model=model,
    tools=[search_tool],
    response_format=ToolStrategy(Output),
)

# Invocation triggers model binding
result = agent.invoke({"messages": [user_message]})

# Internally:
# 1. _get_bound_model creates bound model with output tools
# 2. Model is invoked with bound configuration
# 3. Response is parsed using output bindings
# 4. result["structured_response"] contains Output instance
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_agent_model_binding]]

=== Used By Workflows ===
* Structured_Output_Workflow (Step 4)
