{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/structured-output]]
|-
! Domains
| [[domain::LLMs]], [[domain::Schema_Validation]], [[domain::Type_Safety]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for extracting validated, structured responses from language models using schema definitions and multiple extraction strategies.

=== Description ===
This workflow covers the complete structured output pipeline from schema definition to validated response parsing. The system supports multiple output strategies: tool-based extraction (works with all tool-calling models), provider-native JSON mode (more efficient for supporting models), and automatic strategy selection based on model capabilities.

Key capabilities:
* Multiple schema types: Pydantic models, dataclasses, TypedDict, JSON schema dicts
* Strategy auto-selection based on model profile
* Validation with automatic retry on parse errors
* Union types for multiple valid response schemas
* Custom error handling and retry messages

=== Usage ===
Execute this workflow when you need to:
* Extract structured data from natural language
* Ensure responses conform to a specific schema
* Parse and validate model outputs programmatically
* Build type-safe AI applications

Typical use cases: form filling, entity extraction, API response formatting, data transformation pipelines.

== Execution Steps ==

=== Step 1: Schema Definition ===
[[step::Principle:langchain-ai_langchain_Schema_Definition]]

Define the response schema using Pydantic models, dataclasses, TypedDict, or raw JSON schema dictionaries. The schema determines the structure and types of the extracted response.

'''Supported schema types:'''
* Pydantic `BaseModel`: Full validation, nested models, field descriptions
* `@dataclass`: Simpler syntax, automatic `__init__`
* `TypedDict`: Dictionary with typed keys, no runtime validation
* JSON schema dict: Direct schema specification, no Python type

'''Best practices:'''
* Add docstrings to classes for tool descriptions
* Use `Field()` with descriptions for better extraction
* Keep schemas flat when possible for better model performance

=== Step 2: Strategy Selection ===
[[step::Principle:langchain-ai_langchain_Strategy_Selection]]

Choose or auto-detect the extraction strategy based on model capabilities. The strategy determines how the model produces structured output - either via tool calling or native JSON mode.

'''Available strategies:'''
* `AutoStrategy`: Auto-detect best approach based on model profile (default for raw schemas)
* `ToolStrategy`: Model calls a synthetic tool with schema as arguments
* `ProviderStrategy`: Model outputs JSON directly via provider's native mode

'''Trade-offs:'''
* ToolStrategy: Works with all tool-calling models, supports retry on validation errors
* ProviderStrategy: More efficient (no extra tool call), requires provider support

=== Step 3: Tool Binding (ToolStrategy) ===
[[step::Principle:langchain-ai_langchain_Output_Tool_Binding]]

For tool-based extraction, create synthetic tools from the schema and bind them to the model. The model is forced to call these tools with `tool_choice="any"`, ensuring structured output.

'''What happens:'''
* Schema is converted to `_SchemaSpec` with JSON schema
* `OutputToolBinding` creates `StructuredTool` from schema
* Model is bound with synthetic tools + regular tools
* Union types create multiple tool options

=== Step 4: Model Invocation ===
[[step::Principle:langchain-ai_langchain_Model_Invocation_With_Schema]]

Invoke the model with the configured response format. The model generates output conforming to the schema (via tool call or direct JSON depending on strategy).

'''Invocation flow:'''
1. Messages sent to model with bound tools/format
2. Model returns AIMessage (with tool_calls for ToolStrategy)
3. For ProviderStrategy: JSON parsed from message content
4. For ToolStrategy: Arguments extracted from tool call

=== Step 5: Response Parsing and Validation ===
[[step::Principle:langchain-ai_langchain_Response_Parsing]]

Parse and validate the model output against the schema. The parser converts raw JSON/dict data into the typed Python object defined by the schema.

'''Parsing by schema type:'''
* Pydantic: `TypeAdapter(schema).validate_python(data)`
* Dataclass: `TypeAdapter(schema).validate_python(data)`
* TypedDict: `TypeAdapter(schema).validate_python(data)`
* JSON schema: Return data dict as-is (no Python type)

=== Step 6: Error Handling and Retry ===
[[step::Principle:langchain-ai_langchain_Structured_Output_Error_Handling]]

Handle validation errors with configurable retry behavior. For ToolStrategy, errors can trigger automatic retry by sending an error message back to the model.

'''Error handling options:'''
* `handle_errors=True`: Catch all errors, use default template
* `handle_errors=False`: Propagate exceptions immediately
* `handle_errors="custom message"`: Use custom retry prompt
* `handle_errors=ExceptionType`: Only catch specific exceptions
* `handle_errors=callable`: Custom function `Exception -> str`

'''Error types:'''
* `StructuredOutputValidationError`: Schema validation failed
* `MultipleStructuredOutputsError`: Model called multiple output tools

== Execution Diagram ==
{{#mermaid:graph TD
    A[Schema Definition] --> B[Strategy Selection]
    B --> C{Strategy Type}
    C -->|ToolStrategy| D[Tool Binding]
    C -->|ProviderStrategy| E[Response Format Binding]
    D --> F[Model Invocation]
    E --> F
    F --> G[Response Parsing]
    G --> H{Valid?}
    H -->|Yes| I[Return Typed Response]
    H -->|No| J{Retry Enabled?}
    J -->|Yes| K[Send Error Message]
    K --> F
    J -->|No| L[Raise Exception]
}}

== Related Pages ==
* [[step::Principle:langchain-ai_langchain_Schema_Definition]]
* [[step::Principle:langchain-ai_langchain_Strategy_Selection]]
* [[step::Principle:langchain-ai_langchain_Output_Tool_Binding]]
* [[step::Principle:langchain-ai_langchain_Model_Invocation_With_Schema]]
* [[step::Principle:langchain-ai_langchain_Response_Parsing]]
* [[step::Principle:langchain-ai_langchain_Structured_Output_Error_Handling]]
