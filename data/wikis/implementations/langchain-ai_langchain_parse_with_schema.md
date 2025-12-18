{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Pydantic TypeAdapter|https://docs.pydantic.dev/latest/concepts/type_adapter/]]
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Structured_Output]], [[domain::Data_Validation]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for converting raw model output data into validated, typed instances according to schema definitions, provided by LangChain's structured output system.

=== Description ===

`_parse_with_schema` is the core parsing function that converts dictionary data into typed instances. It uses Pydantic's `TypeAdapter` for validation and respects the `schema_kind` to produce appropriate output types.

Additionally, `OutputToolBinding.parse` and `ProviderStrategyBinding.parse` wrap this function with strategy-specific data extraction logic.

=== Usage ===

Use parsing functions (indirectly via strategy bindings) when:
* Processing tool call arguments from ToolStrategy
* Extracting typed data from JSON mode responses
* Validating arbitrary data against defined schemas

The functions handle Pydantic models, dataclasses, TypedDicts, and raw JSON schemas.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/structured_output.py
* '''Lines:''' L76-101 (_parse_with_schema), L327-339 (OutputToolBinding.parse), L373-401 (ProviderStrategyBinding.parse)

=== Signature ===
<syntaxhighlight lang="python">
def _parse_with_schema(
    schema: type[SchemaT] | dict,
    schema_kind: SchemaKind,
    data: dict[str, Any]
) -> Any:
    """Parse data using for any supported schema type.

    Args:
        schema: The schema type (Pydantic model, dataclass, or TypedDict)
        schema_kind: One of "pydantic", "dataclass", "typeddict", or "json_schema"
        data: The data to parse

    Returns:
        The parsed instance according to the schema type

    Raises:
        ValueError: If parsing fails
    """


class OutputToolBinding:
    def parse(self, tool_args: dict[str, Any]) -> SchemaT:
        """Parse tool arguments according to the schema.

        Args:
            tool_args: The arguments from the tool call

        Returns:
            The parsed response according to the schema type

        Raises:
            ValueError: If parsing fails
        """


class ProviderStrategyBinding:
    def parse(self, response: AIMessage) -> SchemaT:
        """Parse AIMessage content according to the schema.

        Args:
            response: The AIMessage containing the structured output

        Returns:
            The parsed response according to the schema

        Raises:
            ValueError: If text extraction, JSON parsing or schema validation fails
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal functions, used via strategy bindings
from langchain.agents.structured_output import OutputToolBinding, ProviderStrategyBinding

# Through agent execution
agent.invoke(...)  # Parsing happens automatically
</syntaxhighlight>

== I/O Contract ==

=== Inputs (_parse_with_schema) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema || type[SchemaT] | dict || Yes || Schema type or JSON schema dict
|-
| schema_kind || SchemaKind || Yes || "pydantic", "dataclass", "typeddict", or "json_schema"
|-
| data || dict[str, Any] || Yes || Raw data to parse
|}

=== Inputs (OutputToolBinding.parse) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tool_args || dict[str, Any] || Yes || Arguments from model's tool call
|}

=== Inputs (ProviderStrategyBinding.parse) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| response || AIMessage || Yes || Message containing JSON content
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| instance || SchemaT || Validated instance matching schema type
|}

== Usage Examples ==

=== Parsing Pydantic Model ===
<syntaxhighlight lang="python">
from pydantic import BaseModel, Field, TypeAdapter


class Person(BaseModel):
    """Person information."""
    name: str = Field(description="Person's name")
    age: int = Field(ge=0, le=150, description="Age in years")
    email: str | None = None


# Using TypeAdapter directly
data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
adapter = TypeAdapter(Person)
result = adapter.validate_python(data)

print(type(result))  # <class 'Person'>
print(result.name)   # "Alice"
print(result.age)    # 30

# With _parse_with_schema (internal)
from langchain.agents.structured_output import _parse_with_schema

result = _parse_with_schema(Person, "pydantic", data)
print(type(result))  # <class 'Person'>
</syntaxhighlight>

=== Parsing Dataclass ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from pydantic import TypeAdapter


@dataclass
class Product:
    """Product information."""
    name: str
    price: float
    in_stock: bool


data = {"name": "Widget", "price": 29.99, "in_stock": True}
adapter = TypeAdapter(Product)
result = adapter.validate_python(data)

print(type(result))  # <class 'Product'>
print(result.name)   # "Widget"
print(result.price)  # 29.99
</syntaxhighlight>

=== Parsing TypedDict ===
<syntaxhighlight lang="python">
from typing import TypedDict
from pydantic import TypeAdapter


class Config(TypedDict):
    """Application config."""
    host: str
    port: int
    debug: bool


data = {"host": "localhost", "port": 8080, "debug": False}
adapter = TypeAdapter(Config)
result = adapter.validate_python(data)

print(type(result))  # <class 'dict'>
print(result["host"])  # "localhost"
# Result is a dict that matches TypedDict type hint
</syntaxhighlight>

=== Parsing with OutputToolBinding ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import _SchemaSpec, OutputToolBinding
from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


# Create binding
schema_spec = _SchemaSpec(SearchResult)
binding = OutputToolBinding.from_schema_spec(schema_spec)

# Simulate tool call from model
tool_args = {
    "title": "Python Tutorial",
    "url": "https://example.com/python",
    "snippet": "Learn Python programming..."
}

# Parse tool arguments
result = binding.parse(tool_args)

print(type(result))  # <class 'SearchResult'>
print(result.title)  # "Python Tutorial"
</syntaxhighlight>

=== Parsing with ProviderStrategyBinding ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import _SchemaSpec, ProviderStrategyBinding
from langchain_core.messages import AIMessage
from pydantic import BaseModel


class Analysis(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]


# Create binding
schema_spec = _SchemaSpec(Analysis)
binding = ProviderStrategyBinding.from_schema_spec(schema_spec)

# Simulate AI response with JSON content
ai_message = AIMessage(
    content='{"sentiment": "positive", "confidence": 0.95, "keywords": ["great", "excellent"]}'
)

# Parse message content
result = binding.parse(ai_message)

print(type(result))  # <class 'Analysis'>
print(result.sentiment)  # "positive"
print(result.keywords)   # ["great", "excellent"]
</syntaxhighlight>

=== Handling JSON Schema Type ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import _parse_with_schema

# JSON schema: returns raw dict (no validation)
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "number"}
    }
}

data = {"name": "test", "value": 42, "extra": "ignored"}

result = _parse_with_schema(json_schema, "json_schema", data)
print(type(result))  # <class 'dict'>
print(result)  # {"name": "test", "value": 42, "extra": "ignored"}
# Raw dict returned without validation
</syntaxhighlight>

=== Handling Parsing Errors ===
<syntaxhighlight lang="python">
from pydantic import BaseModel, Field, ValidationError
from langchain.agents.structured_output import _SchemaSpec, OutputToolBinding


class StrictSchema(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)  # 0-150 constraint


schema_spec = _SchemaSpec(StrictSchema)
binding = OutputToolBinding.from_schema_spec(schema_spec)

# Valid data
result = binding.parse({"name": "Alice", "age": 30})  # Works

# Invalid data - constraint violation
try:
    result = binding.parse({"name": "Bob", "age": 200})  # age > 150
except ValueError as e:
    print(f"Parsing failed: {e}")
    # "Failed to parse data to StrictSchema: ..."

# Invalid data - missing field
try:
    result = binding.parse({"age": 30})  # missing name
except ValueError as e:
    print(f"Parsing failed: {e}")
</syntaxhighlight>

=== Parsing Complex Nested Structures ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from langchain.agents.structured_output import _SchemaSpec, OutputToolBinding


class Address(BaseModel):
    street: str
    city: str
    country: str


class Company(BaseModel):
    name: str
    address: Address
    employees: list[str]


schema_spec = _SchemaSpec(Company)
binding = OutputToolBinding.from_schema_spec(schema_spec)

data = {
    "name": "TechCorp",
    "address": {
        "street": "123 Main St",
        "city": "San Francisco",
        "country": "USA"
    },
    "employees": ["Alice", "Bob", "Charlie"]
}

result = binding.parse(data)

print(type(result))  # <class 'Company'>
print(type(result.address))  # <class 'Address'>
print(result.address.city)  # "San Francisco"
print(result.employees)  # ["Alice", "Bob", "Charlie"]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Response_Parsing]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
