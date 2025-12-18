{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Structured Outputs|https://platform.openai.com/docs/guides/structured-outputs]]
* [[source::Paper|Pydantic V2|https://docs.pydantic.dev/latest/]]
* [[source::Doc|JSON Schema|https://json-schema.org/]]
|-
! Domains
| [[domain::LLM]], [[domain::Structured_Output]], [[domain::Type_Safety]], [[domain::Schema_Validation]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Design pattern for selecting and configuring methods to extract typed, validated responses from language models.

=== Description ===

Structured Output Strategy is the approach to transforming free-form LLM text generation into type-safe, schema-validated data structures. This principle recognizes that different providers and models support different mechanisms for structured output, and provides a unified abstraction layer.

The core problem solved: LLMs generate text, but applications need structured data. Without structured output, parsing LLM responses is brittle and error-prone. Structured output strategies provide guarantees about response format.

Three main strategies exist:
1. **Tool-based:** Leverage function/tool calling to force structured arguments
2. **Provider-native:** Use provider's JSON mode with schema enforcement
3. **Parsing-based:** Post-process text output with schema validation

The strategy pattern allows applications to select the most appropriate method for their use case and model.

=== Usage ===

Use Structured Output Strategy when:
* Building data extraction pipelines with LLMs
* Creating agents that need typed responses for downstream processing
* Implementing reliable LLM-powered APIs
* Ensuring consistent output format across different models

Strategy selection criteria:
* **Tool-based:** Broad provider support, flexible error handling, works with tool-using models
* **Provider-native:** Strictest validation, best performance, provider-specific
* **Auto:** Cross-provider compatibility, delegates selection to framework

== Theoretical Basis ==

Structured Output Strategy implements the **Strategy Pattern** for output extraction.

'''1. Schema Normalization'''

All schema types are normalized to a common representation:

<syntaxhighlight lang="python">
# Pseudo-code for schema normalization
class SchemaSpec:
    schema: type  # Original schema (Pydantic, dataclass, TypedDict, dict)
    schema_kind: Literal["pydantic", "dataclass", "typeddict", "json_schema"]
    json_schema: dict  # Normalized JSON Schema
    name: str
    description: str

def normalize_schema(schema) -> SchemaSpec:
    if isinstance(schema, dict):
        return SchemaSpec(schema, "json_schema", schema, ...)
    if issubclass(schema, BaseModel):
        return SchemaSpec(schema, "pydantic", schema.model_json_schema(), ...)
    if is_dataclass(schema):
        return SchemaSpec(schema, "dataclass", TypeAdapter(schema).json_schema(), ...)
    if is_typeddict(schema):
        return SchemaSpec(schema, "typeddict", TypeAdapter(schema).json_schema(), ...)
</syntaxhighlight>

'''2. Tool-Based Strategy'''

Creates synthetic tool for structured output:

<syntaxhighlight lang="python">
# Pseudo-code for tool strategy
class ToolStrategy:
    def configure_model(self, model, user_tools):
        # Create output tool from schema
        output_tool = StructuredTool(
            name=self.schema_spec.name,
            description=self.schema_spec.description,
            args_schema=self.schema_spec.json_schema,
        )
        # Bind with tool_choice="any" to force tool use
        return model.bind_tools(
            [*user_tools, output_tool],
            tool_choice="any"
        )

    def parse_response(self, ai_message):
        tool_call = ai_message.tool_calls[-1]  # Get output tool call
        return parse_with_schema(tool_call.args, self.schema_spec)
</syntaxhighlight>

'''3. Provider-Native Strategy'''

Uses OpenAI-style `response_format`:

<syntaxhighlight lang="python">
# Pseudo-code for provider strategy
class ProviderStrategy:
    def configure_model(self, model, user_tools):
        return model.bind_tools(
            user_tools,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema_spec.name,
                    "schema": self.schema_spec.json_schema,
                    "strict": self.strict,
                }
            }
        )

    def parse_response(self, ai_message):
        json_content = json.loads(ai_message.content)
        return parse_with_schema(json_content, self.schema_spec)
</syntaxhighlight>

'''4. Validation and Error Handling'''

Uses Pydantic's TypeAdapter for universal validation:

<syntaxhighlight lang="python">
def parse_with_schema(data: dict, schema_spec: SchemaSpec) -> Any:
    if schema_spec.schema_kind == "json_schema":
        return data  # Raw dict for JSON schema

    # Use Pydantic for validation
    adapter = TypeAdapter(schema_spec.schema)
    return adapter.validate_python(data)  # Raises ValidationError on failure
</syntaxhighlight>

'''5. Error Recovery Pattern'''

Tool strategy supports configurable retry on validation errors:

<syntaxhighlight lang="python">
# Pseudo-code for error handling in agent loop
def handle_structured_output_error(exception, response_format):
    if not response_format.handle_errors:
        raise exception  # No retry

    # Generate error message for model
    if isinstance(response_format.handle_errors, str):
        error_msg = response_format.handle_errors
    elif callable(response_format.handle_errors):
        error_msg = response_format.handle_errors(exception)
    else:
        error_msg = f"Validation failed: {exception}. Please try again."

    # Return (should_retry=True, error_message)
    return True, error_msg
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_ResponseFormat_strategies]]

=== Used By Workflows ===
* Agent_Creation_Workflow (Step 4)
* Structured_Output_Workflow (Step 2)
