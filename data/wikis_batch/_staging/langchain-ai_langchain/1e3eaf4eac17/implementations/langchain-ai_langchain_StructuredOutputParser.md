# StructuredOutputParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.structured`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/structured.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[dict[str, Any]]`

## Overview

`StructuredOutputParser` is a schema-based parser that extracts structured JSON data from LLM responses. It uses `ResponseSchema` definitions to specify expected fields and generates format instructions for the LLM. The parser expects JSON output wrapped in markdown code blocks and validates that all expected keys are present.

### Key Characteristics

- **Output Type**: `dict[str, Any]`
- **Schema-Based**: Uses `ResponseSchema` to define expected structure
- **JSON Format**: Expects JSON in markdown code blocks (` ```json ... ``` `)
- **Type Annotations**: Supports type hints for each field
- **Format Instructions**: Auto-generates clear instructions with examples
- **Validation**: Ensures all expected keys are present in output

## Code Reference

### Class Definition

```python
class ResponseSchema(BaseModel):
    """Schema for a response from a structured output parser."""

    name: str
    """The name of the schema."""
    description: str
    """The description of the schema."""
    type: str = "string"
    """The type of the response."""


class StructuredOutputParser(BaseOutputParser[dict[str, Any]]):
    """Parse the output of an LLM call to a structured output."""

    response_schemas: list[ResponseSchema]
    """The schemas for the response."""

    @classmethod
    def from_response_schemas(
        cls,
        response_schemas: list[ResponseSchema],
    ) -> StructuredOutputParser:
        """Create a StructuredOutputParser from a list of ResponseSchema.

        Args:
            response_schemas: The schemas for the response.

        Returns:
            An instance of StructuredOutputParser.
        """
        return cls(response_schemas=response_schemas)

    def get_format_instructions(
        self,
        only_json: bool = False,
    ) -> str:
        """Get format instructions for the output parser.

        Args:
            only_json: If `True`, only the json in the Markdown code snippet
                will be returned, without the introducing text.
        """
        schema_str = "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas],
        )
        if only_json:
            return STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS.format(format=schema_str)
        return STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema_str)

    @override
    def parse(self, text: str) -> dict[str, Any]:
        expected_keys = [rs.name for rs in self.response_schemas]
        return parse_and_check_json_markdown(text, expected_keys)

    @property
    def _type(self) -> str:
        return "structured"
```

## Input/Output Contract

### ResponseSchema

Defines the structure for each expected field:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Required | Field name (JSON key) |
| `description` | `str` | Required | Human-readable description for LLM guidance |
| `type` | `str` | `"string"` | Type annotation (e.g., "string", "integer", "List[string]") |

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_schemas` | `list[ResponseSchema]` | Required | List of expected field schemas |

### Factory Method: `from_response_schemas()`

Convenient method to create parser from schema list.

**Arguments:**
- `response_schemas` (list[ResponseSchema]): Field schemas

**Returns:**
- `StructuredOutputParser`: Configured parser instance

### Methods

#### `get_format_instructions(only_json: bool = False) -> str`

Generates format instructions for the LLM.

**Arguments:**
- `only_json` (bool): If True, returns only JSON template without explanatory text

**Returns:**
- `str`: Multi-line format instructions with schema and example

**Output Format (only_json=False):**
```
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{
    "field1": type  // description
    "field2": type  // description
}
```
```

#### `parse(text: str) -> dict[str, Any]`

Parses JSON from markdown code block and validates keys.

**Arguments:**
- `text` (str): LLM output containing JSON in markdown

**Returns:**
- `dict[str, Any]`: Parsed JSON object

**Raises:**
- `OutputParserException`: If JSON is invalid or missing expected keys

**Processing:**
1. Extracts JSON from markdown code block
2. Parses JSON string
3. Validates all expected keys are present
4. Returns parsed dictionary

## Examples

### Basic Usage

```python
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema
)

# Define schema
response_schemas = [
    ResponseSchema(name="name", description="Person's name"),
    ResponseSchema(name="age", description="Person's age", type="integer"),
    ResponseSchema(name="city", description="City of residence")
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get format instructions
instructions = parser.get_format_instructions()
print(instructions)
# Output:
# The output should be a markdown code snippet formatted in the following schema...
#
# ```json
# {
#     "name": string  // Person's name
#     "age": integer  // Person's age
#     "city": string  // City of residence
# }
# ```

# Parse LLM response
response = """```json
{
    "name": "Alice",
    "age": 25,
    "city": "Seattle"
}
```"""

result = parser.parse(response)
# Returns: {"name": "Alice", "age": 25, "city": "Seattle"}
```

### Complex Types

```python
response_schemas = [
    ResponseSchema(
        name="title",
        description="Article title",
        type="string"
    ),
    ResponseSchema(
        name="tags",
        description="List of relevant tags",
        type="List[string]"
    ),
    ResponseSchema(
        name="word_count",
        description="Number of words in article",
        type="integer"
    ),
    ResponseSchema(
        name="metadata",
        description="Additional metadata",
        type="dict"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

response = """```json
{
    "title": "Introduction to Python",
    "tags": ["python", "programming", "tutorial"],
    "word_count": 1500,
    "metadata": {"author": "John Doe", "date": "2024-01-15"}
}
```"""

result = parser.parse(response)
# Returns full dictionary with nested structures preserved
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Define schema
response_schemas = [
    ResponseSchema(name="summary", description="Brief summary of the text"),
    ResponseSchema(name="sentiment", description="Overall sentiment (positive/negative/neutral)"),
    ResponseSchema(name="key_points", description="Main points", type="List[string]")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create prompt with format instructions
prompt = PromptTemplate(
    template="""Analyze the following text and provide structured output.

Text: {text}

{format_instructions}""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create chain
chain = prompt | ChatOpenAI() | parser

# Execute
result = chain.invoke({"text": "I love this new product! It's amazing and works perfectly."})
# Returns: {
#     "summary": "Positive product review",
#     "sentiment": "positive",
#     "key_points": ["loves the product", "works perfectly", "thinks it's amazing"]
# }
```

### Only JSON Format

```python
# Get just the JSON schema without explanatory text
parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="answer", description="The answer to the question"),
    ResponseSchema(name="confidence", description="Confidence level (0-100)", type="integer")
])

instructions = parser.get_format_instructions(only_json=True)
print(instructions)
# Output:
# ```json
# {
#     "answer": string  // The answer to the question
#     "confidence": integer  // Confidence level (0-100)
# }
# ```
```

### Data Extraction

```python
response_schemas = [
    ResponseSchema(name="company", description="Company name"),
    ResponseSchema(name="position", description="Job position"),
    ResponseSchema(name="salary_range", description="Salary range"),
    ResponseSchema(name="requirements", description="Key requirements", type="List[string]")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = PromptTemplate(
    template="""Extract structured information from this job posting:

{job_posting}

{format_instructions}""",
    input_variables=["job_posting"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | ChatOpenAI() | parser

result = chain.invoke({"job_posting": "Senior Python Developer at TechCorp. $120k-$150k. Must have 5+ years experience, know Django and FastAPI."})
# Returns: {
#     "company": "TechCorp",
#     "position": "Senior Python Developer",
#     "salary_range": "$120k-$150k",
#     "requirements": ["5+ years experience", "Django", "FastAPI"]
# }
```

### Error Handling

```python
from langchain_core.exceptions import OutputParserException

parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="name", description="Name"),
    ResponseSchema(name="age", description="Age")
])

# Missing expected key
try:
    response = """```json
{
    "name": "Alice"
}
```"""
    parser.parse(response)
except OutputParserException as e:
    print(e)
    # "Missing expected keys in output: ['age']"

# Invalid JSON
try:
    response = """```json
{
    "name": "Alice",
    "age": 25
}
```"""  # Missing comma
    parser.parse(response)
except OutputParserException as e:
    print(e)
    # "Invalid JSON in output: ..."
```

### Multi-Level Structure

```python
response_schemas = [
    ResponseSchema(
        name="user",
        description="User information",
        type="dict"
    ),
    ResponseSchema(
        name="preferences",
        description="User preferences",
        type="dict"
    ),
    ResponseSchema(
        name="status",
        description="Account status"
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

response = """```json
{
    "user": {
        "name": "Alice",
        "email": "alice@example.com"
    },
    "preferences": {
        "theme": "dark",
        "notifications": true
    },
    "status": "active"
}
```"""

result = parser.parse(response)
# Returns nested dictionary structure
# result["user"]["name"] == "Alice"
```

### With OutputFixingParser

```python
from langchain_classic.output_parsers.fix import OutputFixingParser

base_parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="answer", description="The answer"),
    ResponseSchema(name="reasoning", description="Reasoning for the answer")
])

# Wrap with fixing parser for robustness
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI()
)

# Handles malformed JSON automatically
response = """```json
{
    "answer": "Paris"
    "reasoning": "Capital of France"  // Missing comma
}
```"""

result = fixing_parser.parse(response)
# Automatically corrects and returns: {"answer": "Paris", "reasoning": "Capital of France"}
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `YamlOutputParser` - Similar parser for YAML format
- `PydanticOutputParser` - Type-safe parser using Pydantic models
- `OutputFixingParser` - Wrapper for automatic error correction
- `parse_and_check_json_markdown()` - Core parsing function from langchain-core

## Implementation Notes

### Design Decisions

1. **Markdown Code Blocks**: Uses ``` ```json ``` format for clear delineation and syntax highlighting
2. **Schema Objects**: Separates schema definition (ResponseSchema) from parser logic
3. **Type Annotations**: Includes type hints in format instructions to guide LLM
4. **Validation Only**: Validates presence of keys but doesn't enforce types
5. **Flexible Types**: Type field is documentation only (no runtime validation)

### Format Instructions Strategy

The parser generates instructions that:
1. Specify markdown code block requirement
2. Show exact JSON structure with field names
3. Include type annotations for each field
4. Provide inline comments with descriptions
5. Emphasize the need for proper JSON formatting

This approach maximizes LLM compliance with the expected format.

### Parsing Strategy

Uses `parse_and_check_json_markdown()` from langchain-core which:
1. Extracts content between ``` ```json and ``` ```
2. Parses JSON string to Python dict
3. Validates all expected keys are present
4. Returns parsed dictionary

### Type System

The `type` field in ResponseSchema:
- **Documentation Only**: Not enforced at runtime
- **LLM Guidance**: Helps LLM generate appropriate values
- **Common Types**: "string", "integer", "float", "boolean", "List[string]", "dict"
- **No Validation**: Parser doesn't check if values match type annotations

### Performance Considerations

- JSON parsing is fast (native Python `json` module)
- Regex extraction of code blocks is efficient
- Validation is O(n) where n = number of expected keys
- No type checking overhead (documentation only)

### Limitations

1. **No Type Enforcement**: Type annotations are for guidance only
2. **No Default Values**: All fields must be present in output
3. **No Optional Fields**: Cannot specify optional vs. required fields
4. **Flat Validation**: Only validates top-level keys (nested structure not validated)
5. **JSON Only**: Expects JSON format (not other structured formats)
6. **No Schema Validation**: Doesn't validate against JSON Schema spec
7. **No Field Order**: Order of fields doesn't matter but keys must exist

### Comparison with PydanticOutputParser

| Feature | StructuredOutputParser | PydanticOutputParser |
|---------|------------------------|----------------------|
| Type Validation | No (documentation only) | Yes (full Pydantic validation) |
| Schema Definition | ResponseSchema list | Pydantic model |
| Default Values | Not supported | Supported via Pydantic |
| Optional Fields | Not supported | Supported via Pydantic |
| Nested Validation | No | Yes (full model validation) |
| Setup Complexity | Simple | Requires Pydantic model |
| Use Case | Quick structured extraction | Type-safe structured data |

### Use Cases

- Extracting structured information from text
- LLM-powered form filling
- Data normalization and extraction
- Content analysis with structured output
- Information retrieval from documents
- Survey response processing
- Entity extraction with metadata
- Structured summarization

### Best Practices

1. **Clear Descriptions**: Provide detailed descriptions in ResponseSchema for better LLM understanding
2. **Type Hints**: Use type field to guide LLM output format
3. **Validate Downstream**: Add application-level validation for critical fields
4. **Use OutputFixingParser**: Wrap with fixing parser for production robustness
5. **Test Format Instructions**: Verify LLM understands format by testing
6. **Keep Simple**: For complex validation needs, use PydanticOutputParser instead
7. **Handle Errors**: Always wrap parse() in try-except for error handling
8. **Provide Examples**: Include examples in prompts to improve accuracy

### Common Patterns

**Simple Key-Value Extraction:**
```python
ResponseSchema(name="key", description="description")
```

**List Fields:**
```python
ResponseSchema(name="items", description="list of items", type="List[string]")
```

**Numeric Fields:**
```python
ResponseSchema(name="count", description="number of items", type="integer")
ResponseSchema(name="score", description="confidence score", type="float")
```

**Boolean Fields:**
```python
ResponseSchema(name="is_valid", description="validity check", type="boolean")
```

**Nested Objects:**
```python
ResponseSchema(name="metadata", description="additional data", type="dict")
```

### Integration Patterns

**Pattern 1: Progressive Parsing**
```python
# Try simple parser first, fall back to fixing parser
try:
    result = parser.parse(text)
except OutputParserException:
    result = fixing_parser.parse(text)
```

**Pattern 2: Post-Processing**
```python
def parse_and_validate(parser, text, validator_fn):
    result = parser.parse(text)
    validated = validator_fn(result)
    return validated
```

**Pattern 3: Type Coercion**
```python
def parse_with_types(parser, text, type_map):
    result = parser.parse(text)
    for key, type_fn in type_map.items():
        if key in result:
            result[key] = type_fn(result[key])
    return result

# Usage
result = parse_with_types(
    parser,
    text,
    {"age": int, "score": float, "active": bool}
)
```
