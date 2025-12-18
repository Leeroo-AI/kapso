{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Tools|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Structured_Output]], [[domain::Tool_Calling]], [[domain::Type_Safety]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Process of converting a schema specification into a synthetic tool that can be bound to a model for structured output extraction.

=== Description ===

Output Tool Binding bridges schema definitions and model tool calling by creating synthetic tools from output schemas. When using `ToolStrategy`, the schema is converted into a `BaseTool` that the model can "call" to produce structured output.

This binding contains:
* The original schema for type validation
* A `BaseTool` instance for model binding
* A `parse` method to convert tool call arguments back to typed instances

The synthetic tool trick enables structured output on models that may not support native JSON mode.

=== Usage ===

Output Tool Binding occurs automatically when:
* Using `ToolStrategy` for structured output
* Creating agents with `response_format=ToolStrategy(Schema)`
* Model generates tool calls matching the schema

The binding enables:
* Schema-to-tool conversion for model binding
* Tool arguments to typed instance parsing
* Schema kind tracking for proper instantiation

== Theoretical Basis ==

Output Tool Binding implements **Adapter Pattern** to bridge schemas and tool calling.

'''1. Schema to Tool Conversion'''

<syntaxhighlight lang="python">
# SchemaSpec contains normalized schema info
schema_spec = _SchemaSpec(MyPydanticModel)
# - schema: MyPydanticModel
# - name: "MyPydanticModel"
# - description: "Model docstring"
# - json_schema: {...}
# - schema_kind: "pydantic"

# OutputToolBinding creates a tool from the spec
binding = OutputToolBinding.from_schema_spec(schema_spec)
# - binding.schema: MyPydanticModel
# - binding.schema_kind: "pydantic"
# - binding.tool: StructuredTool instance
</syntaxhighlight>

'''2. Binding Data Flow'''

<syntaxhighlight lang="text">
Schema Definition              Tool Binding               Model Invocation
┌─────────────────┐      ┌─────────────────────┐      ┌──────────────────┐
│ class Output    │      │ OutputToolBinding   │      │ model.bind_tools │
│   name: str     │ ──►  │   schema: Output    │ ──►  │   ([tool])       │
│   value: int    │      │   tool: BaseTool    │      │                  │
└─────────────────┘      └─────────────────────┘      └──────────────────┘
                                    │
                                    │ parse()
                                    ▼
                         ┌─────────────────────┐
                         │ Output(             │
                         │   name="x",         │
                         │   value=42          │
                         │ )                   │
                         └─────────────────────┘
</syntaxhighlight>

'''3. StructuredTool Creation'''

<syntaxhighlight lang="python">
from langchain_core.tools import StructuredTool

# Binding creates a StructuredTool from schema spec
def create_output_tool(schema_spec: _SchemaSpec) -> BaseTool:
    """Create a synthetic tool for structured output."""
    return StructuredTool(
        args_schema=schema_spec.json_schema,  # JSON Schema for validation
        name=schema_spec.name,                # Tool name from schema
        description=schema_spec.description,  # Tool description
    )

# The tool appears in model's available tools
# Model "calls" this tool to produce structured output
</syntaxhighlight>

'''4. Multiple Schema Support (Union Types)'''

<syntaxhighlight lang="python">
from typing import Union

class Success(BaseModel):
    status: Literal["success"]
    data: dict

class Error(BaseModel):
    status: Literal["error"]
    message: str

# ToolStrategy handles Union by creating multiple bindings
strategy = ToolStrategy(Union[Success, Error])

# Creates schema_specs for each variant
# strategy.schema_specs = [
#     _SchemaSpec(Success),  # -> OutputToolBinding for Success
#     _SchemaSpec(Error),    # -> OutputToolBinding for Error
# ]

# Model can call either tool depending on response
</syntaxhighlight>

'''5. Parsing Tool Call Results'''

<syntaxhighlight lang="python">
# When model returns a tool call
tool_call = {
    "name": "MyOutput",
    "args": {"name": "result", "value": 42}
}

# Binding parses args to typed instance
binding = output_tool_bindings[tool_call["name"]]
result = binding.parse(tool_call["args"])
# result is MyOutput(name="result", value=42)

# Parsing respects schema_kind
# - "pydantic" -> Pydantic model instance
# - "dataclass" -> dataclass instance
# - "typeddict" -> TypedDict dict
# - "json_schema" -> raw dict
</syntaxhighlight>

'''6. Integration with ToolStrategy'''

<syntaxhighlight lang="python">
# ToolStrategy holds schema_specs
strategy = ToolStrategy(OutputSchema, handle_errors=True)

# Agent factory creates bindings from specs
structured_output_tools = {}
for spec in strategy.schema_specs:
    binding = OutputToolBinding.from_schema_spec(spec)
    structured_output_tools[spec.name] = binding

# Bindings used for:
# 1. Model binding: model.bind_tools([b.tool for b in bindings])
# 2. Response parsing: binding.parse(tool_call.args)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_OutputToolBinding_class]]

=== Used By Workflows ===
* Structured_Output_Workflow (Step 3)
