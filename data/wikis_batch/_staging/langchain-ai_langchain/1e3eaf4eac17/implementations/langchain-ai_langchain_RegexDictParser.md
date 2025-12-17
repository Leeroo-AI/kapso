# RegexDictParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.regex_dict`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/regex_dict.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[dict[str, str]]`

## Overview

`RegexDictParser` is a specialized regex-based parser that extracts multiple independent key-value pairs from text using a pattern template. Unlike `RegexParser` which uses a single regex with capture groups, this parser applies a regex template to multiple expected keys, making it ideal for parsing text with repeated formatting patterns.

### Key Characteristics

- **Output Type**: `dict[str, str]`
- **Template-Based**: Uses single regex pattern template for all keys
- **Multiple Patterns**: Applies pattern independently for each expected key
- **Format Mapping**: Maps output keys to expected format strings
- **No-Update Support**: Optional filtering of specific values
- **Validation**: Ensures exactly one match per key

## Code Reference

### Class Definition

```python
class RegexDictParser(BaseOutputParser[dict[str, str]]):
    """Parse the output of an LLM call into a Dictionary using a regex."""

    regex_pattern: str = r"{}:\s?([^.'\n']*)\.?"
    """The regex pattern to use to parse the output."""
    output_key_to_format: dict[str, str]
    """The keys to use for the output."""
    no_update_value: str | None = None
    """The default key to use for the output."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_dict_parser"

    def parse(self, text: str) -> dict[str, str]:
        """Parse the output of an LLM call."""
        result = {}
        for output_key, expected_format in self.output_key_to_format.items():
            specific_regex = self.regex_pattern.format(re.escape(expected_format))
            matches = re.findall(specific_regex, text)
            if not matches:
                msg = (
                    f"No match found for output key: {output_key} with expected format \
                        {expected_format} on text {text}"
                )
                raise ValueError(msg)
            if len(matches) > 1:
                msg = f"Multiple matches found for output key: {output_key} with \
                        expected format {expected_format} on text {text}"
                raise ValueError(msg)
            if self.no_update_value is not None and matches[0] == self.no_update_value:
                continue
            result[output_key] = matches[0]
        return result
```

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regex_pattern` | `str` | `r"{}:\s?([^.'\n']*)\.?"` | Template pattern with `{}` placeholder |
| `output_key_to_format` | `dict[str, str]` | Required | Maps output keys to expected format labels |
| `no_update_value` | `str \| None` | `None` | Value to skip (won't be added to result) |

### Default Pattern Explanation

The default pattern `r"{}:\s?([^.'\n']*)\.?"` matches:
- `{}` - Placeholder replaced with expected format string
- `:` - Literal colon separator
- `\s?` - Optional whitespace
- `([^.'\n']*)` - Capture group: any characters except period, quote, newline, or quote
- `\.?` - Optional trailing period

### Methods

#### `parse(text: str) -> dict[str, str]`

Parses text by applying pattern template to each expected key.

**Arguments:**
- `text` (str): LLM output to parse

**Returns:**
- `dict[str, str]`: Dictionary with extracted values (excludes no_update_value matches)

**Raises:**
- `ValueError`: If any key has no match or multiple matches

**Process:**
1. Iterate through output_key_to_format items
2. Create specific regex by formatting pattern with expected format
3. Find all matches using `re.findall()`
4. Validate exactly one match exists
5. Skip if value matches no_update_value
6. Add to result dictionary

## Examples

### Basic Key-Value Extraction

```python
from langchain_classic.output_parsers.regex_dict import RegexDictParser

# Parse formatted output
parser = RegexDictParser(
    output_key_to_format={
        "name": "Name",
        "age": "Age",
        "city": "City"
    }
)

text = "Name: Alice. Age: 25. City: Seattle."
result = parser.parse(text)
# Returns: {"name": "Alice", "age": "25", "city": "Seattle"}
```

### Custom Pattern Template

```python
# Use custom pattern with different separator
parser = RegexDictParser(
    regex_pattern=r"{} = (.+?)(?:,|$)",  # Format: "Key = value," or end
    output_key_to_format={
        "status": "Status",
        "priority": "Priority"
    }
)

text = "Status = active, Priority = high"
result = parser.parse(text)
# Returns: {"status": "active", "priority": "high"}
```

### With No-Update Value

```python
# Skip entries with specific value
parser = RegexDictParser(
    output_key_to_format={
        "username": "Username",
        "email": "Email",
        "phone": "Phone"
    },
    no_update_value="N/A"
)

text = "Username: john. Email: N/A. Phone: 555-1234."
result = parser.parse(text)
# Returns: {"username": "john", "phone": "555-1234"}
# Email excluded because value was "N/A"
```

### Product Information Extraction

```python
parser = RegexDictParser(
    output_key_to_format={
        "product": "Product",
        "price": "Price",
        "stock": "Stock",
        "rating": "Rating"
    }
)

text = """
Product: Laptop.
Price: $999.
Stock: 50.
Rating: 4.5.
"""

result = parser.parse(text)
# Returns: {
#     "product": "Laptop",
#     "price": "$999",
#     "stock": "50",
#     "rating": "4.5"
# }
```

### Case-Sensitive Matching

```python
# Parser is case-sensitive by default
parser = RegexDictParser(
    output_key_to_format={
        "title": "Title",
        "author": "Author"
    }
)

# This works
text1 = "Title: Book. Author: Smith."
result = parser.parse(text1)
# Returns: {"title": "Book", "author": "Smith"}

# This fails (lowercase labels)
text2 = "title: Book. author: Smith."
try:
    parser.parse(text2)
except ValueError:
    print("No match - case mismatch")
```

### Different Output Keys vs. Format Labels

```python
# Output keys can differ from format labels
parser = RegexDictParser(
    output_key_to_format={
        "user_name": "Name",       # user_name != Name
        "user_age": "Age",         # user_age != Age
        "user_location": "Location"  # user_location != Location
    }
)

text = "Name: Bob. Age: 30. Location: NYC."
result = parser.parse(text)
# Returns: {
#     "user_name": "Bob",
#     "user_age": "30",
#     "user_location": "NYC"
# }
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = RegexDictParser(
    output_key_to_format={
        "summary": "Summary",
        "sentiment": "Sentiment",
        "confidence": "Confidence"
    }
)

prompt = PromptTemplate(
    template="""Analyze this text: {text}

Provide your analysis in this format:
Summary: <brief summary>
Sentiment: <positive/negative/neutral>
Confidence: <0-100>""",
    input_variables=["text"]
)

chain = prompt | ChatOpenAI() | parser

result = chain.invoke({"text": "I absolutely love this product!"})
# Returns: {
#     "summary": "Positive product review",
#     "sentiment": "positive",
#     "confidence": "95"
# }
```

### Multi-Line Text

```python
parser = RegexDictParser(
    regex_pattern=r"{}:\s*([^\n]+)",  # Match until newline
    output_key_to_format={
        "question": "Question",
        "answer": "Answer",
        "source": "Source"
    }
)

text = """Question: What is Python?
Answer: A programming language
Source: Python documentation"""

result = parser.parse(text)
# Returns: {
#     "question": "What is Python?",
#     "answer": "A programming language",
#     "source": "Python documentation"
# }
```

### Error Handling

```python
parser = RegexDictParser(
    output_key_to_format={
        "name": "Name",
        "age": "Age"
    }
)

# Missing expected key
try:
    parser.parse("Name: Alice.")  # Missing "Age:"
except ValueError as e:
    print(e)
    # "No match found for output key: age with expected format Age on text ..."

# Multiple matches for same key
try:
    parser.parse("Name: Alice. Name: Bob. Age: 25.")
except ValueError as e:
    print(e)
    # "Multiple matches found for output key: name with expected format Name on text ..."
```

### Validation Pattern

```python
# Use pattern that validates format
parser = RegexDictParser(
    regex_pattern=r"{}: (\d{{4}}-\d{{2}}-\d{{2}})",  # ISO date format
    output_key_to_format={
        "start_date": "Start Date",
        "end_date": "End Date"
    }
)

text = "Start Date: 2024-01-01. End Date: 2024-12-31."
result = parser.parse(text)
# Returns: {"start_date": "2024-01-01", "end_date": "2024-12-31"}

# Invalid format fails
try:
    parser.parse("Start Date: Jan 1, 2024. End Date: Dec 31, 2024.")
except ValueError:
    print("Date format validation failed")
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `RegexParser` - Alternative regex parser with single pattern
- `StructuredOutputParser` - Parse structured JSON output
- `BooleanOutputParser` - Parse boolean values
- Python `re` module documentation

## Implementation Notes

### Design Decisions

1. **Template Pattern**: Single regex template applied to multiple keys reduces duplication
2. **Format Escaping**: Uses `re.escape()` on format strings to handle special characters
3. **Exact Match Requirement**: Requires exactly one match per key (strict validation)
4. **No-Update Filtering**: Allows excluding specific sentinel values from results
5. **Independent Matching**: Each key matched independently (no interdependencies)

### Pattern Template System

The template system works by:
1. Pattern contains `{}` placeholder
2. Placeholder replaced with escaped format string for each key
3. Resulting specific pattern used to find matches
4. Captures content after formatted label

**Example:**
- Template: `r"{}:\s?([^.'\n']*)\.?"`
- Format: `"Name"`
- Specific pattern: `r"Name:\s?([^.'\n']*)\.?"`

### Comparison with RegexParser

| Feature | RegexDictParser | RegexParser |
|---------|-----------------|-------------|
| Pattern Type | Template applied to each key | Single regex with groups |
| Matching Strategy | Independent per key | Single pattern match |
| Key Ordering | Order doesn't matter | Must match group order |
| Validation | Strict (exactly one match) | First match only |
| Use Case | Repeated format pattern | Single structured pattern |

### Performance Considerations

- Multiple `re.findall()` calls (one per key)
- Each pattern compiled independently (no pre-compilation)
- Linear time complexity: O(n * m) where n=text length, m=number of keys
- More expensive than RegexParser for large key sets

**Optimization:**
```python
import re

class OptimizedRegexDictParser(RegexDictParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compile all patterns
        self._compiled_patterns = {
            key: re.compile(self.regex_pattern.format(re.escape(fmt)))
            for key, fmt in self.output_key_to_format.items()
        }
```

### Limitations

1. **Exact Match Required**: Must find exactly one match per key (inflexible)
2. **No Optional Keys**: All keys in output_key_to_format must be present
3. **String Output Only**: No type conversion (all values are strings)
4. **Order Independent**: Cannot enforce key ordering
5. **No Nested Structures**: Only supports flat dictionary output
6. **Pattern Shared**: All keys must follow same pattern structure

### Use Cases

- Parsing structured reports with consistent formatting
- Extracting form data from text
- Processing LLM outputs with labeled sections
- Parsing configuration files
- Log file analysis with repeated patterns
- Survey response parsing
- Document metadata extraction

### Best Practices

1. **Test Template**: Verify regex_pattern works with all expected formats
2. **Escape Format Strings**: The parser does this automatically, but be aware
3. **Handle Missing Keys**: Wrap parse() in try-except for optional keys
4. **Validate Values**: Check extracted values meet semantic requirements
5. **Use No-Update Value**: Leverage no_update_value for "N/A" or "Unknown" fields
6. **Document Format**: Clearly specify expected format in LLM prompts
7. **Consider Alternatives**: Use StructuredOutputParser for complex structures

### Common Patterns

**Default Pattern** (key-value with colon):
```python
regex_pattern=r"{}:\s?([^.'\n']*)\.?"
```

**Equals Assignment**:
```python
regex_pattern=r"{}\s*=\s*(.+?)(?:,|$)"
```

**Parentheses**:
```python
regex_pattern=r"{}\(([^)]+)\)"
```

**Quoted Values**:
```python
regex_pattern=r'{}: "([^"]+)"'
```

**Multi-line Values**:
```python
regex_pattern=r"{}:\s*([^\n]+)"
```

### Integration Patterns

**Pattern 1: Optional Keys**
```python
def safe_parse(parser, text, required_keys):
    try:
        result = parser.parse(text)
    except ValueError:
        # Retry with subset of keys
        result = {}
        for key in required_keys:
            # Parse individually
            pass
    return result
```

**Pattern 2: Type Conversion**
```python
def parse_with_types(parser, text, type_map):
    result = parser.parse(text)
    for key, converter in type_map.items():
        if key in result:
            result[key] = converter(result[key])
    return result

# Usage
result = parse_with_types(
    parser,
    text,
    {"age": int, "price": float}
)
```

**Pattern 3: Default Values**
```python
def parse_with_defaults(parser, text, defaults):
    try:
        return parser.parse(text)
    except ValueError:
        # Return defaults for missing keys
        return defaults
```

### Debugging Tips

1. **Test formats individually**: Verify each format string matches
2. **Print specific patterns**: Debug by printing `regex_pattern.format(re.escape(fmt))`
3. **Check escape issues**: Use raw strings `r"..."` for patterns
4. **Verify findall results**: Test `re.findall()` directly on sample text
5. **Test edge cases**: Empty values, special characters, newlines
