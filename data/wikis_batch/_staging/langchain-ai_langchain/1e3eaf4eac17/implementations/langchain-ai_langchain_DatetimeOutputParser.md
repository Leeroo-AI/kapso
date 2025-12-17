# DatetimeOutputParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.datetime`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/datetime.py`
- **Type**: Output Parser
- **Base Class**: `BaseOutputParser[datetime]`

## Overview

`DatetimeOutputParser` is an output parser that converts LLM text responses into Python `datetime` objects. It uses configurable format strings (strftime/strptime patterns) to parse dates and times, and provides intelligent format instructions with examples to help the LLM generate correctly formatted datetime strings.

### Key Characteristics

- **Output Type**: `datetime.datetime`
- **Configurable Format**: Supports any valid strftime format pattern
- **Default Format**: ISO 8601 with microseconds (`%Y-%m-%dT%H:%M:%S.%fZ`)
- **Smart Examples**: Automatically generates format examples based on current time
- **Format Instructions**: Provides clear guidance to LLMs with examples
- **Error Handling**: Raises `OutputParserException` on parsing failures

## Code Reference

### Class Definition

```python
class DatetimeOutputParser(BaseOutputParser[datetime]):
    """Parse the output of an LLM call to a datetime."""

    format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
    """The string value that is used as the datetime format.

    Update this to match the desired datetime format for your application.
    """

    def get_format_instructions(self) -> str:
        """Returns the format instructions for the given format."""
        if self.format == "%Y-%m-%dT%H:%M:%S.%fZ":
            examples = comma_list(
                [
                    "2023-07-04T14:30:00.000000Z",
                    "1999-12-31T23:59:59.999999Z",
                    "2025-01-01T00:00:00.000000Z",
                ],
            )
        else:
            try:
                now = datetime.now(tz=timezone.utc)
                examples = comma_list(
                    [
                        now.strftime(self.format),
                        (now.replace(year=now.year - 1)).strftime(self.format),
                        (now - timedelta(days=1)).strftime(self.format),
                    ],
                )
            except ValueError:
                # Fallback if the format is very unusual
                examples = f"e.g., a valid string in the format {self.format}"

        return (
            f"Write a datetime string that matches the "
            f"following pattern: '{self.format}'.\n\n"
            f"Examples: {examples}\n\n"
            f"Return ONLY this string, no other words!"
        )

    def parse(self, response: str) -> datetime:
        """Parse a string into a datetime object."""
        try:
            return datetime.strptime(response.strip(), self.format)
        except ValueError as e:
            msg = f"Could not parse datetime string: {response}"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
        return "datetime"
```

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str` | `"%Y-%m-%dT%H:%M:%S.%fZ"` | The strftime format pattern for parsing datetime strings |

### Common Format Patterns

| Pattern | Example | Description |
|---------|---------|-------------|
| `%Y-%m-%dT%H:%M:%S.%fZ` | `2023-07-04T14:30:00.000000Z` | ISO 8601 with microseconds |
| `%Y-%m-%d %H:%M:%S` | `2023-07-04 14:30:00` | Standard datetime |
| `%Y-%m-%d` | `2023-07-04` | Date only |
| `%B %d, %Y` | `July 04, 2023` | Human-readable date |
| `%m/%d/%Y` | `07/04/2023` | US date format |
| `%d/%m/%Y` | `04/07/2023` | European date format |

### Methods

#### `get_format_instructions() -> str`

Generates instructions for the LLM with format pattern and examples.

**Returns:**
- `str`: Multi-line instruction string with format pattern and 3 examples

**Example Generation Logic:**
- For default format: Uses hardcoded ISO 8601 examples
- For custom formats: Generates examples using current time, previous year, and yesterday
- Fallback: Returns simple format description if example generation fails

#### `parse(response: str) -> datetime`

Parses a datetime string into a Python datetime object.

**Arguments:**
- `response` (str): LLM output containing datetime string

**Returns:**
- `datetime`: Parsed datetime object (note: timezone-naive by default)

**Raises:**
- `OutputParserException`: If string doesn't match the format pattern

**Processing:**
1. Strips whitespace from input
2. Applies `strptime` with configured format
3. Returns timezone-naive datetime (no timezone conversion)

## Examples

### Basic Usage (Default Format)

```python
from langchain_classic.output_parsers.datetime import DatetimeOutputParser

# Default ISO 8601 format
parser = DatetimeOutputParser()

# Parse ISO 8601 datetime
result = parser.parse("2023-07-04T14:30:00.000000Z")
# Returns: datetime.datetime(2023, 7, 4, 14, 30, 0)

# Get format instructions
instructions = parser.get_format_instructions()
print(instructions)
# Output:
# Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.
#
# Examples: 2023-07-04T14:30:00.000000Z, 1999-12-31T23:59:59.999999Z, 2025-01-01T00:00:00.000000Z
#
# Return ONLY this string, no other words!
```

### Custom Date Format

```python
# US date format
parser = DatetimeOutputParser(format="%m/%d/%Y")

result = parser.parse("07/04/2023")
# Returns: datetime.datetime(2023, 7, 4, 0, 0)

instructions = parser.get_format_instructions()
# Will include current date in MM/DD/YYYY format as examples
```

### Human-Readable Format

```python
# Full month name format
parser = DatetimeOutputParser(format="%B %d, %Y")

result = parser.parse("July 04, 2023")
# Returns: datetime.datetime(2023, 7, 4, 0, 0)
```

### Time Only Format

```python
# Time without date
parser = DatetimeOutputParser(format="%H:%M:%S")

result = parser.parse("14:30:00")
# Returns: datetime.datetime(1900, 1, 1, 14, 30, 0)
# Note: strptime defaults to 1900-01-01 for missing date components
```

### Error Handling

```python
from langchain_core.exceptions import OutputParserException

parser = DatetimeOutputParser()

# Invalid format
try:
    parser.parse("not a date")
except OutputParserException as e:
    print(e)
    # "Could not parse datetime string: not a date"

# Wrong format
try:
    parser.parse("2023-07-04")  # Missing time component
except OutputParserException as e:
    print(e)
    # "Could not parse datetime string: 2023-07-04"
```

### With LLM Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
parser = DatetimeOutputParser()

prompt = PromptTemplate(
    template="""When did the following event occur? {event}

{format_instructions}""",
    input_variables=["event"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

result = chain.invoke({"event": "Moon landing"})
# Returns: datetime.datetime(1969, 7, 20, 20, 17, 0)  (if LLM responds correctly)
```

### European Date Format

```python
# DD/MM/YYYY format
parser = DatetimeOutputParser(format="%d/%m/%Y")

result = parser.parse("04/07/2023")
# Returns: datetime.datetime(2023, 7, 4, 0, 0)
```

### With Timezone Information

```python
# Format with timezone offset
parser = DatetimeOutputParser(format="%Y-%m-%d %H:%M:%S %z")

result = parser.parse("2023-07-04 14:30:00 +0000")
# Returns: datetime.datetime(2023, 7, 4, 14, 30, 0, tzinfo=datetime.timezone.utc)
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `EnumOutputParser` - Parse enum values
- `StructuredOutputParser` - Parse structured JSON output
- `YamlOutputParser` - Parse YAML output
- `OutputFixingParser` - Wrapper that fixes parsing errors with LLM

## Implementation Notes

### Design Decisions

1. **ISO 8601 Default**: Uses widely-accepted international standard as default format
2. **Microsecond Precision**: Default format includes microseconds for maximum precision
3. **Hardcoded Examples**: Default format uses static examples to avoid timezone issues
4. **Dynamic Examples**: Custom formats generate examples from current time for relevance
5. **Timezone Naive**: Returns timezone-naive datetimes by default (unless format includes timezone)
6. **No Auto-Detection**: Requires exact format match; doesn't try multiple formats

### Format Instructions Strategy

The parser uses a three-tier approach for examples:
1. **Default Format**: Hardcoded examples covering different years and edge cases
2. **Custom Format**: Generates examples using current time, previous year, yesterday
3. **Fallback**: Simple text description if example generation fails

This ensures LLMs always receive clear guidance with concrete examples.

### Error Handling

- Wraps Python's `ValueError` from `strptime` into `OutputParserException`
- Includes original input in error message for debugging
- Preserves exception chain with `from e` for traceback

### Performance Considerations

- `strptime` is relatively fast for single datetime parsing
- No caching of format pattern (compiled each time)
- Example generation in `get_format_instructions()` creates datetime objects each time
- For high-volume parsing, consider pre-compiling format or using `dateutil.parser`

### Limitations

1. **No Fuzzy Parsing**: Requires exact format match; won't handle variations
2. **No Timezone Conversion**: Returns datetime in parsed timezone without conversion
3. **No Validation**: Doesn't validate semantic correctness (e.g., future dates)
4. **Single Format**: Can only parse one format per instance
5. **No Relative Dates**: Can't parse "tomorrow", "next week", etc.

### Use Cases

- Event timestamp extraction
- Scheduling and calendar applications
- Historical date parsing
- Log timestamp parsing
- Date range queries
- Appointment booking systems
- Time-sensitive data analysis
