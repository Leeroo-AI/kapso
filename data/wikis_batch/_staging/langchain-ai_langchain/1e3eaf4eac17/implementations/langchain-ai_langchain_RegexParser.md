# RegexParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.regex`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/regex.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[dict[str, str]]`

## Overview

`RegexParser` is a flexible output parser that uses regular expressions to extract structured data from LLM text responses. It captures regex groups and maps them to named output keys, providing a powerful way to parse semi-structured or patterned text without requiring strict JSON formatting.

### Key Characteristics

- **Output Type**: `dict[str, str]`
- **Regex-Based**: Uses Python's `re` module for pattern matching
- **Group Mapping**: Maps regex capture groups to output keys
- **Serializable**: Fully serializable for persistence
- **Fallback Support**: Optional default output key for failed matches
- **Flexible Patterns**: Supports any valid Python regex pattern

## Code Reference

### Class Definition

```python
class RegexParser(BaseOutputParser[dict[str, str]]):
    """Parse the output of an LLM call using a regex."""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    regex: str
    """The regex to use to parse the output."""
    output_keys: list[str]
    """The keys to use for the output."""
    default_output_key: str | None = None
    """The default key to use for the output."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_parser"

    def parse(self, text: str) -> dict[str, str]:
        """Parse the output of an LLM call."""
        match = re.search(self.regex, text)
        if match:
            return {key: match.group(i + 1) for i, key in enumerate(self.output_keys)}
        if self.default_output_key is None:
            msg = f"Could not parse output: {text}"
            raise ValueError(msg)
        return {
            key: text if key == self.default_output_key else ""
            for key in self.output_keys
        }
```

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regex` | `str` | Required | Regular expression pattern with capture groups |
| `output_keys` | `list[str]` | Required | Names for each capture group (in order) |
| `default_output_key` | `str \| None` | `None` | Key to receive full text if regex doesn't match |

### Methods

#### `parse(text: str) -> dict[str, str]`

Parses text using regex pattern and returns captured groups as dictionary.

**Arguments:**
- `text` (str): LLM output to parse

**Returns:**
- `dict[str, str]`: Dictionary mapping output keys to captured group values

**Raises:**
- `ValueError`: If regex doesn't match and no default_output_key is set

**Behavior:**
- If regex matches: Returns dictionary with each key mapped to its capture group
- If regex fails and default_output_key is set: Returns dictionary with full text in default key, empty strings for others
- If regex fails and no default_output_key: Raises ValueError

## Examples

### Basic Pattern Matching

```python
from langchain_classic.output_parsers.regex import RegexParser

# Parse name and age from text
parser = RegexParser(
    regex=r"Name: (\w+), Age: (\d+)",
    output_keys=["name", "age"]
)

result = parser.parse("Name: Alice, Age: 25")
# Returns: {"name": "Alice", "age": "25"}
```

### Email Extraction

```python
# Extract email components
parser = RegexParser(
    regex=r"(\w+)@(\w+)\.(\w+)",
    output_keys=["username", "domain", "tld"]
)

result = parser.parse("Contact me at john@example.com")
# Returns: {"username": "john", "domain": "example", "tld": "com"}
```

### Date Parsing

```python
# Parse date components
parser = RegexParser(
    regex=r"(\d{4})-(\d{2})-(\d{2})",
    output_keys=["year", "month", "day"]
)

result = parser.parse("The date is 2024-03-15.")
# Returns: {"year": "2024", "month": "03", "day": "15"}
```

### Price Extraction

```python
# Extract price and currency
parser = RegexParser(
    regex=r"(\$|€|£)([\d,]+\.?\d*)",
    output_keys=["currency", "amount"]
)

result = parser.parse("The total cost is $1,234.56")
# Returns: {"currency": "$", "amount": "1,234.56"}
```

### With Default Fallback

```python
# Use default key when pattern doesn't match
parser = RegexParser(
    regex=r"Answer: (\w+)",
    output_keys=["answer"],
    default_output_key="answer"
)

# Successful match
result = parser.parse("Answer: Yes")
# Returns: {"answer": "Yes"}

# Failed match - falls back to entire text
result = parser.parse("I'm not sure")
# Returns: {"answer": "I'm not sure"}
```

### Phone Number Parsing

```python
# Parse phone number components
parser = RegexParser(
    regex=r"\((\d{3})\)\s*(\d{3})-(\d{4})",
    output_keys=["area_code", "exchange", "number"]
)

result = parser.parse("Call me at (555) 123-4567")
# Returns: {
#     "area_code": "555",
#     "exchange": "123",
#     "number": "4567"
# }
```

### Multi-Line Text

```python
# Parse structured response
parser = RegexParser(
    regex=r"Title: (.+)\nAuthor: (.+)\nYear: (\d{4})",
    output_keys=["title", "author", "year"]
)

text = """Title: The Great Gatsby
Author: F. Scott Fitzgerald
Year: 1925"""

result = parser.parse(text)
# Returns: {
#     "title": "The Great Gatsby",
#     "author": "F. Scott Fitzgerald",
#     "year": "1925"
# }
```

### URL Parsing

```python
# Extract URL components
parser = RegexParser(
    regex=r"(https?)://([^/]+)(/.*)?",
    output_keys=["protocol", "domain", "path"]
)

result = parser.parse("Visit https://example.com/page")
# Returns: {
#     "protocol": "https",
#     "domain": "example.com",
#     "path": "/page"
# }
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = RegexParser(
    regex=r"Sentiment: (\w+), Confidence: ([\d.]+)",
    output_keys=["sentiment", "confidence"]
)

prompt = PromptTemplate(
    template="""Analyze the sentiment of this text: {text}

Respond in the format: Sentiment: <positive/negative/neutral>, Confidence: <0.0-1.0>""",
    input_variables=["text"]
)

chain = prompt | ChatOpenAI() | parser

result = chain.invoke({"text": "I love this product!"})
# Returns: {"sentiment": "positive", "confidence": "0.95"}
```

### Error Handling

```python
# Without default key - raises error on no match
parser = RegexParser(
    regex=r"Result: (\d+)",
    output_keys=["result"]
)

try:
    parser.parse("No result found")
except ValueError as e:
    print(e)
    # "Could not parse output: No result found"

# With default key - graceful fallback
parser_with_fallback = RegexParser(
    regex=r"Result: (\d+)",
    output_keys=["result"],
    default_output_key="result"
)

result = parser_with_fallback.parse("No result found")
# Returns: {"result": "No result found"}
```

### Complex Pattern

```python
# Parse structured data with multiple groups
parser = RegexParser(
    regex=r"Product: (.+?) \| Price: \$(\d+\.?\d*) \| Rating: ([\d.]+)/5",
    output_keys=["product", "price", "rating"]
)

result = parser.parse("Product: Laptop | Price: $999.99 | Rating: 4.5/5")
# Returns: {
#     "product": "Laptop",
#     "price": "999.99",
#     "rating": "4.5"
# }
```

### Case-Insensitive Matching

```python
# Use regex flags via (?i) prefix
parser = RegexParser(
    regex=r"(?i)status: (active|inactive)",
    output_keys=["status"]
)

result = parser.parse("Status: ACTIVE")
# Returns: {"status": "ACTIVE"}
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `RegexDictParser` - Alternative regex parser with different interface
- `StructuredOutputParser` - Parse structured JSON output
- `BooleanOutputParser` - Parse boolean values
- Python `re` module documentation

## Implementation Notes

### Design Decisions

1. **Single Match**: Uses `re.search()` to find first match (not `findall()`)
2. **Group Indexing**: Maps capture groups 1-N to output keys (group 0 is full match)
3. **String Values**: All captured groups returned as strings (no type conversion)
4. **Fallback Strategy**: Default key receives entire input if pattern fails
5. **No Pre-compilation**: Regex is compiled on each parse call

### Regex Pattern Requirements

- Must use **capture groups** `()` for extraction
- Number of capture groups must match number of output keys
- Group 0 (full match) is not included in output
- Non-capturing groups `(?:...)` don't count toward output keys

### Performance Considerations

- `re.search()` stops at first match (efficient)
- Regex compilation happens on each call (consider pre-compiling for high-volume)
- Simple patterns are very fast
- Complex patterns with backtracking can be slow

**Optimization for High-Volume:**
```python
import re

# Pre-compile regex
compiled_pattern = re.compile(r"Name: (\w+)")

class OptimizedRegexParser(RegexParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled = re.compile(self.regex)

    def parse(self, text: str):
        match = self._compiled.search(text)
        # ... rest of logic
```

### Limitations

1. **String Output Only**: All values returned as strings (no automatic type conversion)
2. **First Match Only**: Only captures first occurrence (won't find multiple matches)
3. **Fixed Key Count**: Number of keys must match capture groups
4. **No Group Names**: Doesn't support named capture groups like `(?P<name>...)`
5. **No Validation**: Doesn't validate captured content semantically
6. **Order Dependent**: Keys must be in same order as capture groups

### Comparison with RegexDictParser

| Feature | RegexParser | RegexDictParser |
|---------|-------------|-----------------|
| Pattern | Single regex | Pattern template per key |
| Matching | One pattern | Multiple sub-patterns |
| Fallback | Default key | No update value |
| Use Case | Single structured pattern | Multiple independent patterns |

### Use Cases

- Parsing semi-structured text (logs, reports)
- Extracting specific patterns (emails, URLs, dates)
- Converting natural language to structured data
- Parsing LLM outputs with consistent formatting
- Quick prototyping without strict JSON requirements
- Legacy system integration (parsing fixed formats)
- Error messages and log parsing

### Best Practices

1. **Test Patterns**: Validate regex patterns with test data before deployment
2. **Use Raw Strings**: Always use `r"..."` for regex patterns to avoid escape issues
3. **Non-Greedy Matching**: Use `.*?` instead of `.*` to prevent over-matching
4. **Anchors**: Use `^` and `$` for start/end matching when needed
5. **Optional Groups**: Use `(...)?` for optional components
6. **Whitespace**: Use `\s*` to handle variable whitespace
7. **Fallback Keys**: Always set `default_output_key` for production robustness
8. **Validation**: Validate parsed values downstream (regex doesn't guarantee semantics)

### Common Patterns

**Key-Value Pairs:**
```python
regex=r"(\w+): ([^,\n]+)"
```

**Quoted Strings:**
```python
regex=r'"([^"]+)"'
```

**Numbers:**
```python
regex=r"([-+]?\d*\.?\d+)"  # Integer or decimal
```

**Dates (ISO):**
```python
regex=r"(\d{4})-(\d{2})-(\d{2})"
```

**Optional Sections:**
```python
regex=r"Required: (\w+)(?: Optional: (\w+))?"
```

### Debugging Tips

1. **Test patterns online**: Use regex101.com or similar tools
2. **Print raw strings**: Debug escaping issues
3. **Check group count**: Ensure groups match output_keys length
4. **Test edge cases**: Empty strings, special characters, multi-line
5. **Use verbose mode**: `(?x)` flag for complex patterns
