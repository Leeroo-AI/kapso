# EnumOutputParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.enum`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/enum.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[Enum]`

## Overview

`EnumOutputParser` is an output parser that validates and converts LLM text responses into Python Enum values. It ensures the output is one of a predefined set of valid values, making it ideal for classification tasks, multiple-choice questions, and any scenario requiring selection from a fixed set of options.

### Key Characteristics

- **Output Type**: `Enum` (Python enum type)
- **String Values Only**: Requires enum values to be strings
- **Strict Validation**: Only accepts exact matches from valid values
- **Type Safety**: Returns proper Enum instances, not strings
- **Clear Instructions**: Provides comma-separated list of valid options
- **Error Handling**: Raises `OutputParserException` with expected values on mismatch

## Code Reference

### Class Definition

```python
class EnumOutputParser(BaseOutputParser[Enum]):
    """Parse an output that is one of a set of values."""

    enum: type[Enum]
    """The enum to parse. Its values must be strings."""

    @pre_init
    def _raise_deprecation(cls, values: dict) -> dict:
        enum = values["enum"]
        if not all(isinstance(e.value, str) for e in enum):
            msg = "Enum values must be strings"
            raise ValueError(msg)
        return values

    @property
    def _valid_values(self) -> list[str]:
        return [e.value for e in self.enum]

    @override
    def parse(self, response: str) -> Enum:
        try:
            return self.enum(response.strip())
        except ValueError as e:
            msg = (
                f"Response '{response}' is not one of the "
                f"expected values: {self._valid_values}"
            )
            raise OutputParserException(msg) from e

    @override
    def get_format_instructions(self) -> str:
        return f"Select one of the following options: {', '.join(self._valid_values)}"

    @property
    @override
    def OutputType(self) -> type[Enum]:
        return self.enum
```

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enum` | `type[Enum]` | Required | Python Enum class with string values |

### Validation Rules

1. **String Values**: All enum values must be strings (validated at initialization)
2. **Exact Match**: Response must exactly match one enum value (after stripping whitespace)
3. **Case Sensitive**: Matching is case-sensitive by default

### Methods

#### `get_format_instructions() -> str`

Generates instructions listing all valid enum values.

**Returns:**
- `str`: Comma-separated list of valid options

**Example Output:**
```
Select one of the following options: POSITIVE, NEGATIVE, NEUTRAL
```

#### `parse(response: str) -> Enum`

Parses and validates LLM output against enum values.

**Arguments:**
- `response` (str): LLM output containing enum value

**Returns:**
- `Enum`: Matching enum instance

**Raises:**
- `OutputParserException`: If response doesn't match any valid value

**Processing:**
1. Strips whitespace from response
2. Attempts to create enum instance with value
3. Returns enum instance if successful
4. Raises exception with valid values if not found

#### `OutputType` Property

Returns the enum class type for type checking and introspection.

**Returns:**
- `type[Enum]`: The enum class used by this parser

## Examples

### Basic Usage

```python
from enum import Enum
from langchain_classic.output_parsers.enum import EnumOutputParser

# Define an enum
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

# Create parser
parser = EnumOutputParser(enum=Sentiment)

# Get format instructions
instructions = parser.get_format_instructions()
print(instructions)
# Output: "Select one of the following options: positive, negative, neutral"

# Parse valid response
result = parser.parse("positive")
print(result)  # Sentiment.POSITIVE
print(type(result))  # <enum 'Sentiment'>
print(result.value)  # "positive"
```

### Classification Task

```python
from enum import Enum

class Category(str, Enum):
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    POLITICS = "politics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"

parser = EnumOutputParser(enum=Category)

# Parse response
result = parser.parse("technology")
# Returns: Category.TECHNOLOGY

# Use enum for logic
if result == Category.TECHNOLOGY:
    print("Tech article detected")
```

### Priority Levels

```python
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

parser = EnumOutputParser(enum=Priority)

result = parser.parse("high")
# Returns: Priority.HIGH

# Enum comparison works
if result.value == "high":
    print("High priority task")
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

llm = ChatOpenAI()
parser = EnumOutputParser(enum=Sentiment)

prompt = PromptTemplate(
    template="""Analyze the sentiment of the following text: {text}

{format_instructions}""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

result = chain.invoke({"text": "I love this product!"})
# Returns: Sentiment.POSITIVE
```

### Multiple Choice Question

```python
from enum import Enum

class Answer(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

parser = EnumOutputParser(enum=Answer)

prompt = PromptTemplate(
    template="""Question: {question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

{format_instructions}""",
    input_variables=["question", "option_a", "option_b", "option_c", "option_d"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

result = chain.invoke({
    "question": "What is the capital of France?",
    "option_a": "London",
    "option_b": "Berlin",
    "option_c": "Paris",
    "option_d": "Madrid"
})
# Returns: Answer.C
```

### Error Handling

```python
from langchain_core.exceptions import OutputParserException
from enum import Enum

class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

parser = EnumOutputParser(enum=Status)

# Invalid value
try:
    parser.parse("unknown")
except OutputParserException as e:
    print(e)
    # "Response 'unknown' is not one of the expected values: ['active', 'inactive', 'pending']"

# Case sensitive (this will fail)
try:
    parser.parse("Active")  # Should be "active"
except OutputParserException as e:
    print(e)
    # "Response 'Active' is not one of the expected values: ['active', 'inactive', 'pending']"
```

### With OutputFixingParser

```python
from langchain_classic.output_parsers.fix import OutputFixingParser
from langchain_openai import ChatOpenAI
from enum import Enum

class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

base_parser = EnumOutputParser(enum=Color)
llm = ChatOpenAI()

# Wrap with fixing parser to handle errors
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm
)

# Will attempt to fix if LLM returns invalid value
result = fixing_parser.parse("reddish")  # Invalid, but fixing parser will retry
```

### Custom String Enum

```python
from enum import Enum

class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

parser = EnumOutputParser(enum=Environment)

result = parser.parse("production")
# Returns: Environment.PRODUCTION

# Access enum attributes
print(result.name)   # "PRODUCTION"
print(result.value)  # "production"
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `BooleanOutputParser` - Parse boolean values (binary classification)
- `StructuredOutputParser` - Parse structured JSON output
- `OutputFixingParser` - Wrapper that fixes parsing errors with LLM
- `RegexParser` - Regex-based parsing for pattern matching

## Implementation Notes

### Design Decisions

1. **String Values Only**: Restricts enums to string values for LLM compatibility
2. **Pre-initialization Validation**: Uses `@pre_init` to validate enum values before instance creation
3. **Type Preservation**: Returns actual Enum instances, not strings, for type safety
4. **Whitespace Handling**: Strips whitespace to handle common LLM formatting variations
5. **Explicit Error Messages**: Includes list of valid values in error messages for easy debugging

### Validation Strategy

The parser validates at two stages:
1. **Initialization**: Checks all enum values are strings (raises `ValueError`)
2. **Parsing**: Validates response matches enum value (raises `OutputParserException`)

This ensures type safety and prevents runtime errors from invalid configurations.

### Type Safety Benefits

Returning Enum instances provides:
- IDE autocomplete for enum members
- Type checking with mypy/pyright
- Exhaustiveness checking in match/case statements
- Prevents typos in downstream code
- Clear contract for valid values

### Performance Considerations

- Enum value lookup is O(1) using hash-based lookup
- `_valid_values` property creates list on each access (not cached)
- String stripping is minimal overhead
- No regex or complex parsing needed

### Limitations

1. **Case Sensitive**: No built-in case-insensitive matching
2. **No Fuzzy Matching**: Requires exact match; won't handle typos
3. **String Values Only**: Cannot use integer or other enum value types
4. **No Aliases**: Each enum value must be unique
5. **No Partial Matching**: Won't match substring or approximate values

### Use Cases

- Sentiment analysis (positive/negative/neutral)
- Text classification (category selection)
- Multiple choice questions
- Status tracking (active/inactive/pending)
- Priority levels (low/medium/high/critical)
- Environment selection (dev/staging/prod)
- Language detection
- Product categories
- User role classification
- Content moderation (approve/reject/review)

### Best Practices

1. **Use uppercase for enum names**: `POSITIVE` not `positive` (Python convention)
2. **Use lowercase for values**: `"positive"` not `"POSITIVE"` (easier for LLMs)
3. **Keep values simple**: Short, clear strings work best
4. **Provide context**: Use descriptive enum class names
5. **Consider case insensitivity**: Implement custom logic if needed for robustness
