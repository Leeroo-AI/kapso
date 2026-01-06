{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|LangChain Tools|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Structured_Output]], [[domain::Tool_Calling]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for converting schema specifications into synthetic tools for structured output extraction, provided by LangChain's structured output system.

=== Description ===

`OutputToolBinding` is a dataclass that bridges schema definitions and model tool calling. It contains:
* The original schema (Pydantic model, dataclass, TypedDict, or JSON dict)
* Schema kind classification for proper parsing
* A `BaseTool` instance (StructuredTool) for model binding

The `from_schema_spec` factory method creates bindings from `_SchemaSpec` instances, and the `parse` method converts tool call arguments back to typed instances.

=== Usage ===

Use `OutputToolBinding` (indirectly via ToolStrategy) when:
* Extracting structured output via tool calling
* Building agents with typed responses
* Working with Union types that need multiple output tools

The binding is created automatically when using `ToolStrategy` in agent configuration.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/structured_output.py
* '''Lines:''' L289-339

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class OutputToolBinding(Generic[SchemaT]):
    """Information for tracking structured output tool metadata.

    This contains all necessary information to handle structured responses
    generated via tool calls, including the original schema, its type classification,
    and the corresponding tool implementation used by the tools strategy.
    """

    schema: type[SchemaT]
    """The original schema provided for structured output
    (Pydantic model, dataclass, TypedDict, or JSON schema dict)."""

    schema_kind: SchemaKind
    """Classification of the schema type for proper response construction."""

    tool: BaseTool
    """LangChain tool instance created from the schema for model binding."""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create an OutputToolBinding instance from a SchemaSpec.

        Args:
            schema_spec: The SchemaSpec to convert

        Returns:
            An OutputToolBinding instance with the appropriate tool created
        """

    def parse(self, tool_args: dict[str, Any]) -> SchemaT:
        """Parse tool arguments according to the schema.

        Args:
            tool_args: The arguments from the tool call

        Returns:
            The parsed response according to the schema type

        Raises:
            ValueError: If parsing fails
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal class, typically not imported directly
# Used via ToolStrategy in agent configuration
from langchain.agents.structured_output import ToolStrategy

# Bindings created internally when using ToolStrategy
strategy = ToolStrategy(MySchema)
# Agent factory creates OutputToolBinding from strategy.schema_specs
</syntaxhighlight>

== I/O Contract ==

=== Inputs (from_schema_spec) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema_spec || _SchemaSpec[SchemaT] || Yes || Schema specification with schema, name, description, json_schema
|}

=== Inputs (parse) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tool_args || dict[str, Any] || Yes || Arguments from model's tool call
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| binding || OutputToolBinding[SchemaT] || from_schema_spec: Binding with schema, schema_kind, and tool
|-
| instance || SchemaT || parse: Typed instance according to schema
|}

== Usage Examples ==

=== Creating Binding from SchemaSpec ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import _SchemaSpec, OutputToolBinding
from pydantic import BaseModel, Field


class WeatherResponse(BaseModel):
    """Weather information."""
    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions")
    humidity: int = Field(description="Humidity percentage")


# Create schema spec
schema_spec = _SchemaSpec(WeatherResponse)

# Create binding from spec
binding = OutputToolBinding.from_schema_spec(schema_spec)

print(f"Schema: {binding.schema}")  # WeatherResponse
print(f"Kind: {binding.schema_kind}")  # "pydantic"
print(f"Tool name: {binding.tool.name}")  # "WeatherResponse"
</syntaxhighlight>

=== Parsing Tool Call Arguments ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import _SchemaSpec, OutputToolBinding
from pydantic import BaseModel


class ExtractedData(BaseModel):
    name: str
    age: int
    email: str


schema_spec = _SchemaSpec(ExtractedData)
binding = OutputToolBinding.from_schema_spec(schema_spec)

# Simulate tool call args from model
tool_args = {
    "name": "John Doe",
    "age": 30,
    "email": "john@example.com"
}

# Parse to typed instance
result = binding.parse(tool_args)

print(type(result))  # <class 'ExtractedData'>
print(result.name)   # "John Doe"
print(result.age)    # 30
print(result.email)  # "john@example.com"
</syntaxhighlight>

=== Binding with Dataclass Schema ===
<syntaxhighlight lang="python">
from dataclasses import dataclass as dc
from langchain.agents.structured_output import _SchemaSpec, OutputToolBinding


@dc
class Person:
    """A person record."""
    name: str
    age: int
    city: str | None = None


schema_spec = _SchemaSpec(Person)
binding = OutputToolBinding.from_schema_spec(schema_spec)

print(f"Kind: {binding.schema_kind}")  # "dataclass"

# Parse returns dataclass instance
result = binding.parse({"name": "Alice", "age": 25, "city": "NYC"})
print(type(result))  # <class 'Person'>
</syntaxhighlight>

=== Binding with TypedDict Schema ===
<syntaxhighlight lang="python">
from typing import TypedDict
from langchain.agents.structured_output import _SchemaSpec, OutputToolBinding


class Config(TypedDict):
    """Application configuration."""
    host: str
    port: int
    debug: bool


schema_spec = _SchemaSpec(Config)
binding = OutputToolBinding.from_schema_spec(schema_spec)

print(f"Kind: {binding.schema_kind}")  # "typeddict"

# Parse returns TypedDict-compliant dict
result = binding.parse({"host": "localhost", "port": 8080, "debug": True})
print(type(result))  # <class 'dict'>
print(result["host"])  # "localhost"
</syntaxhighlight>

=== Multiple Bindings for Union Types ===
<syntaxhighlight lang="python">
from typing import Union, Literal
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy, _SchemaSpec, OutputToolBinding


class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str


# ToolStrategy creates specs for each variant
strategy = ToolStrategy(Union[SuccessResponse, ErrorResponse])

# Create bindings for each schema spec
bindings = {
    spec.name: OutputToolBinding.from_schema_spec(spec)
    for spec in strategy.schema_specs
}

# bindings = {
#     "SuccessResponse": OutputToolBinding(...),
#     "ErrorResponse": OutputToolBinding(...)
# }

# Model can call either tool
success_result = bindings["SuccessResponse"].parse({"status": "success", "data": {}})
error_result = bindings["ErrorResponse"].parse({"status": "error", "message": "Failed"})
</syntaxhighlight>

=== Using Binding Tool for Model Binding ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import _SchemaSpec, OutputToolBinding
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class Output(BaseModel):
    answer: str
    confidence: float


schema_spec = _SchemaSpec(Output)
binding = OutputToolBinding.from_schema_spec(schema_spec)

# Bind tool to model
model = init_chat_model("gpt-4o")
bound_model = model.bind_tools([binding.tool], tool_choice="any")

# Invoke model - it will "call" the Output tool
response = bound_model.invoke("What is 2+2? Answer with confidence.")

# Parse the tool call
if response.tool_calls:
    result = binding.parse(response.tool_calls[0]["args"])
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Output_Tool_Binding]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
