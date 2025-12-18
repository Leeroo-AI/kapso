{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Pydantic|https://docs.pydantic.dev/]]
* [[source::Doc|JSON Schema|https://json-schema.org/]]
* [[source::Doc|TypedDict|https://docs.python.org/3/library/typing.html#typing.TypedDict]]
|-
! Domains
| [[domain::LLM]], [[domain::Type_Safety]], [[domain::Schema_Validation]], [[domain::Data_Modeling]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Process of specifying the structure and constraints of expected LLM outputs using Python type systems or JSON Schema.

=== Description ===

Schema Definition is the first step in structured output extraction, where you define what the LLM's response should look like. Python's rich type system provides multiple ways to express schemas:
* **Pydantic models:** Full validation, default values, complex types
* **Dataclasses:** Lightweight, stdlib, type hints
* **TypedDict:** Dict-like with typed keys
* **JSON Schema:** Maximum flexibility, interop with other systems

The schema serves dual purposes:
1. Tells the LLM what format to produce
2. Validates and parses the response into typed objects

=== Usage ===

Define schemas when:
* Building structured data extraction pipelines
* Creating typed API responses from LLMs
* Implementing function calling outputs
* Ensuring consistent output format across runs

Schema choice guidelines:
* **Pydantic:** Default choice, best validation and IDE support
* **Dataclass:** When avoiding Pydantic dependency
* **TypedDict:** For dict-first workflows
* **JSON Schema:** For dynamic/generated schemas

== Theoretical Basis ==

Schema Definition implements **Type-Driven Development** for LLM outputs.

'''1. Schema Formats'''

<syntaxhighlight lang="python">
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import TypedDict

# 1. Pydantic (recommended)
class PydanticSchema(BaseModel):
    """Pydantic provides rich validation."""
    name: str = Field(description="Person's name")
    age: int = Field(ge=0, le=150, description="Age in years")
    tags: list[str] = Field(default_factory=list)


# 2. Dataclass
@dataclass
class DataclassSchema:
    """Dataclasses are lightweight."""
    name: str
    age: int
    tags: list[str] | None = None


# 3. TypedDict
class TypedDictSchema(TypedDict):
    """TypedDict for dict-like access."""
    name: str
    age: int
    tags: list[str]


# 4. JSON Schema (dict)
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Person's name"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "tags": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age"]
}
</syntaxhighlight>

'''2. JSON Schema Generation'''

<syntaxhighlight lang="python">
from pydantic import TypeAdapter

def generate_json_schema(schema):
    """Convert any schema type to JSON Schema."""
    if isinstance(schema, dict):
        return schema  # Already JSON Schema

    if hasattr(schema, "model_json_schema"):  # Pydantic
        return schema.model_json_schema()

    # Dataclass or TypedDict
    return TypeAdapter(schema).json_schema()
</syntaxhighlight>

'''3. Schema Kind Detection'''

<syntaxhighlight lang="python">
from dataclasses import is_dataclass
from typing_extensions import is_typeddict
from pydantic import BaseModel

def classify_schema(schema) -> str:
    """Determine schema kind for proper handling."""
    if isinstance(schema, dict):
        return "json_schema"
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return "pydantic"
    if is_dataclass(schema):
        return "dataclass"
    if is_typeddict(schema):
        return "typeddict"
    raise ValueError(f"Unsupported schema type: {type(schema)}")
</syntaxhighlight>

'''4. Description Extraction'''

<syntaxhighlight lang="python">
def extract_description(schema) -> str:
    """Extract description for LLM context."""
    if isinstance(schema, dict):
        return schema.get("description", "")

    # Use docstring
    return getattr(schema, "__doc__", "") or ""

def extract_name(schema) -> str:
    """Extract schema name."""
    if isinstance(schema, dict):
        return schema.get("title", "response_format")

    return getattr(schema, "__name__", "response_format")
</syntaxhighlight>

'''5. Field Descriptions'''

<syntaxhighlight lang="python">
from pydantic import BaseModel, Field

# Descriptions help the LLM understand what to generate
class WellDocumented(BaseModel):
    """Response containing weather information for a location."""

    temperature: float = Field(
        description="Current temperature in degrees Celsius"
    )
    conditions: str = Field(
        description="Weather conditions (e.g., 'sunny', 'cloudy', 'rainy')"
    )
    humidity: int = Field(
        ge=0, le=100,
        description="Relative humidity as a percentage (0-100)"
    )
    wind_speed: float | None = Field(
        default=None,
        description="Wind speed in km/h, if available"
    )
</syntaxhighlight>

'''6. Complex Type Support'''

<syntaxhighlight lang="python">
from pydantic import BaseModel
from typing import Union, Literal
from enum import Enum


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(BaseModel):
    """A task with nested structure."""
    title: str
    priority: Priority
    tags: list[str]
    metadata: dict[str, str]


class TaskList(BaseModel):
    """Multiple tasks."""
    tasks: list[Task]
    total_count: int


# Union types for polymorphic output
class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: dict


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str


Response = Union[SuccessResponse, ErrorResponse]
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_SchemaSpec_class]]

=== Used By Workflows ===
* Structured_Output_Workflow (Step 1)
