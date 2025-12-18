{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Pydantic TypeAdapter|https://docs.pydantic.dev/latest/concepts/type_adapter/]]
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|JSON Parsing|https://docs.python.org/3/library/json.html]]
|-
! Domains
| [[domain::LLM]], [[domain::Structured_Output]], [[domain::Type_Safety]], [[domain::Data_Validation]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Process of converting raw model output (tool call arguments or JSON content) into validated, typed instances according to the schema definition.

=== Description ===

Response Parsing is the final transformation step in structured output extraction. It takes raw data from the model and produces typed Python objects:

* **ToolStrategy:** Parses tool call arguments via `OutputToolBinding.parse`
* **ProviderStrategy:** Parses message content via `ProviderStrategyBinding.parse`

The core parsing logic in `_parse_with_schema` handles all schema kinds (Pydantic, dataclass, TypedDict, JSON schema) using Pydantic's `TypeAdapter` for validation.

=== Usage ===

Response parsing occurs when:
* Processing structured output from tool calls
* Extracting typed data from JSON mode responses
* Validating LLM output against defined schemas

The parsing respects `schema_kind` to produce the correct instance type.

== Theoretical Basis ==

Response Parsing implements **Polymorphic Deserialization** based on schema kind.

'''1. Core Parsing Function'''

<syntaxhighlight lang="python">
from pydantic import TypeAdapter

def _parse_with_schema(
    schema: type[SchemaT] | dict,
    schema_kind: SchemaKind,  # "pydantic" | "dataclass" | "typeddict" | "json_schema"
    data: dict[str, Any]
) -> Any:
    """Parse data using any supported schema type."""
    # JSON schema: return raw data
    if schema_kind == "json_schema":
        return data

    # All other types: use TypeAdapter for validation
    try:
        adapter: TypeAdapter[SchemaT] = TypeAdapter(schema)
        return adapter.validate_python(data)
    except Exception as e:
        schema_name = getattr(schema, "__name__", str(schema))
        raise ValueError(f"Failed to parse data to {schema_name}: {e}") from e
</syntaxhighlight>

'''2. ToolStrategy Parsing Flow'''

<syntaxhighlight lang="text">
Model Response              Tool Call             Parsed Instance
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ AIMessage with  │      │ tool_call:      │      │ MySchema(       │
│ tool_calls: [   │ ──►  │   name: "My..." │ ──►  │   field1=...,   │
│   {...}         │      │   args: {...}   │      │   field2=...    │
│ ]               │      │                 │      │ )               │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                               │
                               │ OutputToolBinding.parse(args)
                               │
                         ┌─────▼─────────────────┐
                         │ _parse_with_schema(   │
                         │   schema, kind, args  │
                         │ )                     │
                         └───────────────────────┘
</syntaxhighlight>

'''3. ProviderStrategy Parsing Flow'''

<syntaxhighlight lang="text">
Model Response              JSON Content          Parsed Instance
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ AIMessage with  │      │ {"field1": ..., │      │ MySchema(       │
│ content: "{...}"│ ──►  │  "field2": ...} │ ──►  │   field1=...,   │
│                 │      │                 │      │   field2=...    │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                               │
                               │ json.loads() + _parse_with_schema()
                               │
                         ┌─────▼─────────────────────┐
                         │ ProviderStrategyBinding   │
                         │   .parse(response)        │
                         └───────────────────────────┘
</syntaxhighlight>

'''4. Schema Kind Polymorphism'''

<syntaxhighlight lang="python">
# TypeAdapter handles all schema kinds uniformly
from pydantic import TypeAdapter, BaseModel
from dataclasses import dataclass
from typing import TypedDict

# Pydantic model
class PydanticSchema(BaseModel):
    name: str
    value: int

# Dataclass
@dataclass
class DataclassSchema:
    name: str
    value: int

# TypedDict
class TypedDictSchema(TypedDict):
    name: str
    value: int

# All parse the same way
data = {"name": "test", "value": 42}

pydantic_result = TypeAdapter(PydanticSchema).validate_python(data)
# Returns: PydanticSchema(name="test", value=42)

dataclass_result = TypeAdapter(DataclassSchema).validate_python(data)
# Returns: DataclassSchema(name="test", value=42)

typeddict_result = TypeAdapter(TypedDictSchema).validate_python(data)
# Returns: {"name": "test", "value": 42} (TypedDict-compliant dict)
</syntaxhighlight>

'''5. AIMessage Content Extraction'''

<syntaxhighlight lang="python">
# ProviderStrategyBinding extracts text from AIMessage
def _extract_text_content_from_message(message: AIMessage) -> str:
    """Extract text content from an AIMessage."""
    content = message.content

    # Simple string
    if isinstance(content, str):
        return content

    # List of content blocks
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and "text" in c:
                    parts.append(str(c["text"]))
                elif "content" in c and isinstance(c["content"], str):
                    parts.append(c["content"])
            else:
                parts.append(str(c))
        return "".join(parts)

    return str(content)
</syntaxhighlight>

'''6. Error Handling in Parsing'''

<syntaxhighlight lang="python">
# Parsing can fail for various reasons
try:
    result = binding.parse(tool_args)
except ValueError as e:
    # Validation error
    # - Missing required fields
    # - Type mismatch
    # - Constraint violation
    pass

# With ToolStrategy, errors can trigger retry
# (see Structured_Output_Error_Handling)

# With ProviderStrategy, errors typically propagate
# (provider enforced the schema)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_parse_with_schema]]

=== Used By Workflows ===
* Structured_Output_Workflow (Step 5)
