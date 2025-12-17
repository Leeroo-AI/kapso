# CombiningOutputParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.combining`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/combining.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[dict[str, Any]]`

## Overview

`CombiningOutputParser` is a meta-parser that combines multiple output parsers into a single parser. It processes text containing multiple distinct outputs separated by double newlines, applying each parser to its corresponding section and merging the results into a single dictionary. This allows LLMs to produce multiple different types of structured outputs in a single response.

### Key Characteristics

- **Output Type**: `dict[str, Any]`
- **Minimum Parsers**: Requires at least 2 parsers
- **Serializable**: Fully serializable for persistence
- **Sequential Processing**: Applies parsers in order to text sections
- **Dictionary Merging**: Combines results from all parsers
- **Validation**: Prevents nesting and list parser combinations

## Code Reference

### Class Definition

```python
class CombiningOutputParser(BaseOutputParser[dict[str, Any]]):
    """Combine multiple output parsers into one."""

    parsers: list[BaseOutputParser]

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @pre_init
    def validate_parsers(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate the parsers."""
        parsers = values["parsers"]
        if len(parsers) < _MIN_PARSERS:
            msg = "Must have at least two parsers"
            raise ValueError(msg)
        for parser in parsers:
            if parser._type == "combining":
                msg = "Cannot nest combining parsers"
                raise ValueError(msg)
            if parser._type == "list":
                msg = "Cannot combine list parsers"
                raise ValueError(msg)
        return values

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "combining"

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        initial = f"For your first output: {self.parsers[0].get_format_instructions()}"
        subsequent = "\n".join(
            f"Complete that output fully. Then produce another output, separated by two newline characters: {p.get_format_instructions()}"
            for p in self.parsers[1:]
        )
        return f"{initial}\n{subsequent}"

    def parse(self, text: str) -> dict[str, Any]:
        """Parse the output of an LLM call."""
        texts = text.split("\n\n")
        output = {}
        for txt, parser in zip(texts, self.parsers, strict=False):
            output.update(parser.parse(txt.strip()))
        return output
```

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parsers` | `list[BaseOutputParser]` | Required | List of output parsers to combine (minimum 2) |

### Validation Rules

1. **Minimum Count**: Must have at least 2 parsers
2. **No Nesting**: Cannot include another `CombiningOutputParser`
3. **No List Parsers**: Cannot include list-type parsers

### Methods

#### `get_format_instructions() -> str`

Generates combined format instructions for the LLM.

**Returns:**
- `str`: Instructions describing how to format multiple outputs with double newline separators

#### `parse(text: str) -> dict[str, Any]`

Parses text containing multiple outputs into a single dictionary.

**Arguments:**
- `text` (str): LLM output containing multiple sections separated by `\n\n`

**Returns:**
- `dict[str, Any]`: Merged dictionary containing results from all parsers

### Parsing Logic

1. Split input text by double newlines (`\n\n`)
2. Iterate through text sections and parsers simultaneously
3. Apply each parser to its corresponding text section
4. Merge all parser outputs using `dict.update()`
5. Return combined dictionary

## Examples

### Basic Usage

```python
from langchain_classic.output_parsers.combining import CombiningOutputParser
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema
)
from langchain_classic.output_parsers.boolean import BooleanOutputParser

# Define individual parsers
answer_parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="answer", description="answer to the question"),
    ResponseSchema(name="source", description="source of information")
])

confidence_parser = BooleanOutputParser(true_val="HIGH", false_val="LOW")

# Combine parsers
combined_parser = CombiningOutputParser(parsers=[answer_parser, confidence_parser])

# Get format instructions
instructions = combined_parser.get_format_instructions()
print(instructions)
# Output:
# For your first output: ...structured format...
# Complete that output fully. Then produce another output, separated by two newline characters: ...boolean format...
```

### Parsing Combined Output

```python
# LLM response with multiple sections
llm_output = """```json
{
    "answer": "Paris is the capital of France",
    "source": "World Geography Database"
}
```

HIGH"""

result = combined_parser.parse(llm_output)
# Returns: {
#     "answer": "Paris is the capital of France",
#     "source": "World Geography Database",
#     True: True
# }
```

### Multiple Structured Outputs

```python
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema
)

# Parser for main analysis
analysis_parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="summary", description="brief summary"),
    ResponseSchema(name="key_points", description="main takeaways")
])

# Parser for metadata
metadata_parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="confidence", description="confidence level"),
    ResponseSchema(name="sources", description="number of sources")
])

# Combine them
combined = CombiningOutputParser(parsers=[analysis_parser, metadata_parser])

text = """```json
{
    "summary": "Climate change impacts coastal regions",
    "key_points": "Rising sea levels, increased storms"
}
```

```json
{
    "confidence": "high",
    "sources": "12"
}
```"""

result = combined.parse(text)
# Returns: {
#     "summary": "Climate change impacts coastal regions",
#     "key_points": "Rising sea levels, increased storms",
#     "confidence": "high",
#     "sources": "12"
# }
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

prompt = PromptTemplate(
    template="""Answer the question and provide metadata.

Question: {question}

{format_instructions}""",
    input_variables=["question"],
    partial_variables={"format_instructions": combined_parser.get_format_instructions()}
)

chain = prompt | llm | combined_parser

result = chain.invoke({"question": "What is the capital of France?"})
# Returns merged dictionary with all parsed fields
```

### Error Handling

```python
# Validation errors
try:
    # Too few parsers
    parser = CombiningOutputParser(parsers=[answer_parser])
except ValueError as e:
    print(e)  # "Must have at least two parsers"

try:
    # Nested combining parser
    nested = CombiningOutputParser(parsers=[
        answer_parser,
        CombiningOutputParser(parsers=[metadata_parser, confidence_parser])
    ])
except ValueError as e:
    print(e)  # "Cannot nest combining parsers"
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `StructuredOutputParser` - Parse structured JSON output
- `BooleanOutputParser` - Parse boolean values
- `YamlOutputParser` - Parse YAML output
- `RegexParser` - Regex-based parsing

## Implementation Notes

### Design Decisions

1. **Double Newline Separator**: Uses `\n\n` as the delimiter between outputs, which is a natural separator in text
2. **Dictionary Merging**: Uses `update()` to merge results, meaning later parsers can override earlier keys
3. **Strict=False Zipping**: Uses non-strict zip to handle mismatched lengths gracefully
4. **Pre-initialization Validation**: Uses `@pre_init` decorator to validate parsers before instantiation

### Validation Logic

The parser prevents:
- **Nesting**: Combining parsers cannot contain other combining parsers (prevents complexity)
- **List Parsers**: List-type parsers are excluded (incompatible output format)
- **Single Parser**: Requires at least 2 parsers (otherwise, just use the single parser directly)

### Format Instructions

The format instructions are dynamically generated by:
1. Taking the first parser's instructions as the starting point
2. Appending each subsequent parser's instructions with separator guidance
3. Emphasizing the need to complete each output fully before starting the next

### Performance Considerations

- Sequential processing means parsers are applied in order
- No parallelization of parser execution
- Dictionary merging is efficient but can lead to key conflicts
- Text splitting by `\n\n` is simple but may be fragile if LLM doesn't follow format exactly

### Limitations

1. **Key Conflicts**: If multiple parsers produce the same keys, later parsers overwrite earlier ones
2. **Fixed Separator**: Relies on exact `\n\n` separator; extra whitespace could cause issues
3. **No Partial Matching**: If text has fewer sections than parsers, remaining parsers are skipped
4. **Order Dependent**: Parser order matters due to sequential application and dictionary updating

### Use Cases

- Combining structured analysis with metadata
- Generating content plus validation information
- Multi-aspect analysis (content + sentiment + confidence)
- Collecting different data types from single LLM call
- Building complex output schemas from simpler components
