{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Pydantic|https://docs.pydantic.dev/]]
* [[source::Doc|JSON Schema|https://json-schema.org/]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Schema_Validation]], [[domain::Type_Safety]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for normalizing different schema formats (Pydantic, dataclass, TypedDict, JSON) into a common internal representation, provided by LangChain's structured output system.

=== Description ===

`_SchemaSpec` is an internal dataclass that standardizes schema representation across different Python type systems. It accepts:
* Pydantic `BaseModel` subclasses
* Python dataclasses
* TypedDict classes
* Raw JSON schema dictionaries

And produces a normalized representation with:
* Schema kind classification
* Generated JSON schema
* Name and description extraction
* Optional strict mode flag

=== Usage ===

Use `_SchemaSpec` (indirectly via strategy classes) when:
* Defining structured output schemas
* Creating tool schemas from Pydantic models
* Working with multiple schema formats in the same system

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/structured_output.py
* '''Lines:''' L104-177

=== Signature ===
<syntaxhighlight lang="python">
@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""

    schema: type[SchemaT]
    """The schema (Pydantic model, dataclass, TypedDict, or JSON schema dict)."""

    name: str
    """Name of the schema (from class name or JSON schema title)."""

    description: str
    """Description (from docstring or JSON schema description)."""

    schema_kind: SchemaKind
    """Kind: 'pydantic', 'dataclass', 'typeddict', or 'json_schema'."""

    json_schema: dict[str, Any]
    """Generated JSON schema representation."""

    strict: bool | None = None
    """Whether to enforce strict validation."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> None:
        """Initialize SchemaSpec from any supported schema type.

        Args:
            schema: Pydantic model, dataclass, TypedDict, or JSON schema dict
            name: Override name (defaults to class name or schema title)
            description: Override description (defaults to docstring)
            strict: Enable strict validation mode
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal class, typically not imported directly
# Used via ToolStrategy, ProviderStrategy, etc.
from langchain.agents.structured_output import ToolStrategy

# Schema passed to strategy is wrapped in _SchemaSpec internally
strategy = ToolStrategy(MyPydanticModel)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| schema || type[SchemaT] || Yes || Schema type to normalize
|-
| name || str | None || No || Override schema name
|-
| description || str | None || No || Override schema description
|-
| strict || bool | None || No || Enable strict validation
|}

=== Outputs (Attributes) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| schema || type[SchemaT] || Original schema reference
|-
| name || str || Resolved name
|-
| description || str || Resolved description
|-
| schema_kind || SchemaKind || "pydantic", "dataclass", "typeddict", or "json_schema"
|-
| json_schema || dict || Generated JSON schema
|-
| strict || bool | None || Strict mode flag
|}

== Usage Examples ==

=== Pydantic Model ===
<syntaxhighlight lang="python">
from pydantic import BaseModel, Field
from langchain.agents.structured_output import _SchemaSpec


class WeatherResponse(BaseModel):
    """Response containing weather information."""

    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions")
    humidity: int = Field(description="Humidity percentage")


spec = _SchemaSpec(WeatherResponse)
print(f"Name: {spec.name}")  # "WeatherResponse"
print(f"Kind: {spec.schema_kind}")  # "pydantic"
print(f"Description: {spec.description}")  # "Response containing..."
print(f"JSON Schema: {spec.json_schema}")
# {
#   "type": "object",
#   "properties": {
#     "temperature": {"type": "number", "description": "Temperature in Celsius"},
#     ...
#   }
# }
</syntaxhighlight>

=== Dataclass ===
<syntaxhighlight lang="python">
from dataclasses import dataclass
from langchain.agents.structured_output import _SchemaSpec


@dataclass
class Person:
    """A person record."""
    name: str
    age: int
    email: str | None = None


spec = _SchemaSpec(Person)
print(f"Kind: {spec.schema_kind}")  # "dataclass"
print(f"JSON Schema: {spec.json_schema}")
</syntaxhighlight>

=== TypedDict ===
<syntaxhighlight lang="python">
from typing import TypedDict
from langchain.agents.structured_output import _SchemaSpec


class Config(TypedDict):
    """Application configuration."""
    host: str
    port: int
    debug: bool


spec = _SchemaSpec(Config)
print(f"Kind: {spec.schema_kind}")  # "typeddict"
</syntaxhighlight>

=== Raw JSON Schema ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import _SchemaSpec

json_schema = {
    "title": "SearchResult",
    "description": "A search result item",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "url": {"type": "string"},
        "snippet": {"type": "string"}
    },
    "required": ["title", "url"]
}

spec = _SchemaSpec(json_schema)
print(f"Name: {spec.name}")  # "SearchResult"
print(f"Kind: {spec.schema_kind}")  # "json_schema"
print(f"JSON Schema: {spec.json_schema}")  # Same as input
</syntaxhighlight>

=== Name and Description Override ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from langchain.agents.structured_output import _SchemaSpec


class InternalModel(BaseModel):
    """Internal model docstring."""
    value: str


# Override name and description
spec = _SchemaSpec(
    InternalModel,
    name="PublicAPI",
    description="Externally facing API response"
)
print(f"Name: {spec.name}")  # "PublicAPI" (overridden)
print(f"Description: {spec.description}")  # "Externally facing..." (overridden)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Schema_Definition]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
