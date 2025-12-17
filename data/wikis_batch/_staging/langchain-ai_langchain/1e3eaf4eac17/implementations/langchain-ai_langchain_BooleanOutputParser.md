# BooleanOutputParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.boolean`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/boolean.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[bool]`

## Overview

`BooleanOutputParser` is an output parser that extracts boolean values from LLM text responses. It searches for configurable true/false string values within the text and returns a Python boolean. The parser uses regex pattern matching with word boundaries to find the boolean indicators, handling case-insensitive matching and detecting ambiguous responses.

### Key Characteristics

- **Output Type**: `bool`
- **Configurable Values**: Custom strings for true/false representation (default: "YES"/"NO")
- **Pattern Matching**: Uses word boundary regex for precise matching
- **Case Insensitive**: Matches values regardless of case
- **Ambiguity Detection**: Raises error if both true and false values appear in text
- **Error Handling**: Provides clear error messages when values are missing or ambiguous

## Code Reference

### Class Definition

```python
class BooleanOutputParser(BaseOutputParser[bool]):
    """Parse the output of an LLM call to a boolean."""

    true_val: str = "YES"
    """The string value that should be parsed as True."""
    false_val: str = "NO"
    """The string value that should be parsed as False."""

    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean.

        Args:
            text: output of a language model

        Returns:
            boolean
        """
        regexp = rf"\b({self.true_val}|{self.false_val})\b"

        truthy = {
            val.upper()
            for val in re.findall(regexp, text, flags=re.IGNORECASE | re.MULTILINE)
        }
        if self.true_val.upper() in truthy:
            if self.false_val.upper() in truthy:
                msg = (
                    f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                    f"in received: {text}."
                )
                raise ValueError(msg)
            return True
        if self.false_val.upper() in truthy:
            if self.true_val.upper() in truthy:
                msg = (
                    f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                    f"in received: {text}."
                )
                raise ValueError(msg)
            return False
        msg = (
            f"BooleanOutputParser expected output value to include either "
            f"{self.true_val} or {self.false_val}. Received {text}."
        )
        raise ValueError(msg)

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "boolean_output_parser"
```

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `true_val` | `str` | `"YES"` | The string value that should be parsed as True |
| `false_val` | `str` | `"NO"` | The string value that should be parsed as False |

### Methods

#### `parse(text: str) -> bool`

Parses the output of an LLM call to a boolean value.

**Arguments:**
- `text` (str): Output text from a language model

**Returns:**
- `bool`: True if true_val found, False if false_val found

**Raises:**
- `ValueError`: If both values are present (ambiguous) or neither value is found

### Pattern Matching Logic

1. Creates regex pattern with word boundaries: `\b(true_val|false_val)\b`
2. Searches text using case-insensitive, multiline flags
3. Collects all matches and converts to uppercase
4. Checks for true_val presence (returns True if no false_val)
5. Checks for false_val presence (returns False if no true_val)
6. Raises ValueError if both or neither are found

## Examples

### Basic Usage

```python
from langchain_classic.output_parsers.boolean import BooleanOutputParser

# Default YES/NO parser
parser = BooleanOutputParser()

# Parse affirmative response
result = parser.parse("The answer is YES.")
# Returns: True

# Parse negative response
result = parser.parse("No, this is not correct.")
# Returns: False
```

### Custom Boolean Values

```python
# Custom true/false strings
parser = BooleanOutputParser(true_val="CORRECT", false_val="INCORRECT")

result = parser.parse("The statement is CORRECT based on the evidence.")
# Returns: True

result = parser.parse("This is INCORRECT according to the data.")
# Returns: False
```

### Error Handling

```python
parser = BooleanOutputParser()

# Ambiguous response
try:
    parser.parse("YES, but also NO in some cases")
except ValueError as e:
    print(e)
    # "Ambiguous response. Both YES and NO in received: ..."

# Missing boolean value
try:
    parser.parse("I'm not sure about this.")
except ValueError as e:
    print(e)
    # "BooleanOutputParser expected output value to include either YES or NO. Received ..."
```

### Case Insensitive Matching

```python
parser = BooleanOutputParser()

# All these work (case insensitive)
parser.parse("yes")      # True
parser.parse("YES")      # True
parser.parse("Yes")      # True
parser.parse("no")       # False
parser.parse("NO")       # False
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
parser = BooleanOutputParser()

prompt = PromptTemplate(
    template="Is the following statement true? {statement}\nAnswer YES or NO.",
    input_variables=["statement"]
)

chain = prompt | llm | parser

result = chain.invoke({"statement": "The sky is blue"})
# Returns: True or False
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `EnumOutputParser` - Parse output to enum values
- `StructuredOutputParser` - Parse structured JSON output
- `RegexParser` - Generic regex-based parsing
- `OutputFixingParser` - Wrapper that fixes parsing errors with LLM

## Implementation Notes

### Design Decisions

1. **Word Boundaries**: Uses `\b` regex boundaries to ensure exact word matching and avoid false positives (e.g., "YESTERDAY" won't match "YES")
2. **Case Insensitive**: All comparisons are done in uppercase to handle various casing
3. **Ambiguity Detection**: Explicit check for both values prevents unclear results
4. **Multiline Support**: Regex flags include MULTILINE for text spanning multiple lines

### Performance Considerations

- Regex compilation happens on each parse call (could be optimized by pre-compiling)
- Simple regex pattern ensures fast matching
- Set operations for efficient duplicate removal

### Error Messages

The parser provides three types of error messages:
1. Ambiguous responses when both values are found
2. Missing values when neither is found
3. Each includes the original text for debugging

### Use Cases

- Yes/no question answering
- Binary classification tasks
- Sentiment analysis (positive/negative)
- Validation checks (valid/invalid)
- Decision making (approve/reject)
