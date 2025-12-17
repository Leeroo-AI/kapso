# OutputFixingParser

## Metadata
- **Package**: `langchain-classic`
- **Module**: `langchain_classic.output_parsers.fix`
- **Source File**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/output_parsers/fix.py`
- **Type**: Output Parser Wrapper
- **Base Class**: `BaseOutputParser[T]` (Generic)

## Overview

`OutputFixingParser` is a wrapper parser that adds automatic error correction to any existing output parser. When the wrapped parser fails, it uses an LLM to fix the malformed output by providing the original completion, format instructions, and error message. This implements a retry-with-correction pattern that significantly improves robustness when dealing with unpredictable LLM outputs.

### Key Characteristics

- **Output Type**: `T` (Generic - matches wrapped parser)
- **Wrapper Pattern**: Wraps any `BaseOutputParser`
- **Retry Logic**: Configurable number of retry attempts
- **LLM-Powered Fixing**: Uses another LLM call to correct errors
- **Serializable**: Fully serializable for persistence
- **Async Support**: Includes async parsing methods
- **Backward Compatible**: Supports both legacy and modern runnable chains

## Code Reference

### Class Definition

```python
class OutputFixingParser(BaseOutputParser[T]):
    """Wrap a parser and try to fix parsing errors."""

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    parser: Annotated[Any, SkipValidation()]
    """The parser to use to parse the output."""
    retry_chain: Annotated[
        RunnableSerializable[OutputFixingParserRetryChainInput, str] | Any,
        SkipValidation(),
    ]
    """The RunnableSerializable to use to retry the completion (Legacy: LLMChain)."""
    max_retries: int = 1
    """The maximum number of times to retry the parse."""
    legacy: bool = True
    """Whether to use the run or arun method of the retry_chain."""

    @classmethod
    def from_llm(
        cls,
        llm: Runnable,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_FIX_PROMPT,
        max_retries: int = 1,
    ) -> OutputFixingParser[T]:
        """Create an OutputFixingParser from a language model and a parser.

        Args:
            llm: llm to use for fixing
            parser: parser to use for parsing
            prompt: prompt to use for fixing
            max_retries: Maximum number of retries to parse.

        Returns:
            OutputFixingParser
        """
        chain = prompt | llm | StrOutputParser()
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    @override
    def parse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "run"):
                    completion = self.retry_chain.run(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    try:
                        completion = self.retry_chain.invoke(
                            {
                                "instructions": self.parser.get_format_instructions(),
                                "completion": completion,
                                "error": repr(e),
                            },
                        )
                    except (NotImplementedError, AttributeError):
                        # Case: self.parser does not have get_format_instructions
                        completion = self.retry_chain.invoke(
                            {
                                "completion": completion,
                                "error": repr(e),
                            },
                        )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    async def aparse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise
                retries += 1
                if self.legacy and hasattr(self.retry_chain, "arun"):
                    completion = await self.retry_chain.arun(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )
                else:
                    try:
                        completion = await self.retry_chain.ainvoke(
                            {
                                "instructions": self.parser.get_format_instructions(),
                                "completion": completion,
                                "error": repr(e),
                            },
                        )
                    except (NotImplementedError, AttributeError):
                        # Case: self.parser does not have get_format_instructions
                        completion = await self.retry_chain.ainvoke(
                            {
                                "completion": completion,
                                "error": repr(e),
                            },
                        )

        msg = "Failed to parse"
        raise OutputParserException(msg)

    @override
    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "output_fixing"

    @property
    @override
    def OutputType(self) -> type[T]:
        return self.parser.OutputType
```

## Input/Output Contract

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parser` | `BaseOutputParser[T]` | Required | The base parser to wrap |
| `retry_chain` | `Runnable` | Required | Chain/LLM to use for fixing errors |
| `max_retries` | `int` | `1` | Maximum number of retry attempts |
| `legacy` | `bool` | `True` | Whether to use legacy run/arun methods |

### Factory Method: `from_llm()`

Convenient method to create an OutputFixingParser with default configuration.

**Arguments:**
- `llm` (Runnable): Language model to use for fixing
- `parser` (BaseOutputParser[T]): Parser to wrap
- `prompt` (BasePromptTemplate): Prompt template for fixing (default: NAIVE_FIX_PROMPT)
- `max_retries` (int): Maximum retries (default: 1)

**Returns:**
- `OutputFixingParser[T]`: Configured fixing parser

### Methods

#### `parse(completion: str) -> T`

Parses completion with automatic error fixing.

**Arguments:**
- `completion` (str): LLM output to parse

**Returns:**
- `T`: Parsed output matching wrapped parser's type

**Raises:**
- `OutputParserException`: If parsing fails after all retry attempts

**Algorithm:**
1. Try parsing with base parser
2. On OutputParserException:
   - If max retries reached, raise exception
   - Call retry_chain with instructions, completion, and error
   - Update completion with fixed version
   - Retry parse (increment retry counter)
3. Repeat until success or max retries

#### `aparse(completion: str) -> T` (Async)

Async version of parse with same behavior.

#### `get_format_instructions() -> str`

Delegates to wrapped parser's format instructions.

#### `OutputType` Property

Returns the output type of the wrapped parser.

## Examples

### Basic Usage

```python
from langchain_classic.output_parsers.fix import OutputFixingParser
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema
)
from langchain_openai import ChatOpenAI

# Base parser that might fail
base_parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="name", description="Person's name"),
    ResponseSchema(name="age", description="Person's age", type="integer")
])

# Wrap with fixing parser
llm = ChatOpenAI()
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm,
    max_retries=2
)

# This malformed JSON will be automatically fixed
malformed = """```json
{
    "name": "John"
    "age": "thirty"  # Missing comma, age is string not int
}
```"""

result = fixing_parser.parse(malformed)
# The LLM will fix the JSON and retry parsing
# Returns: {"name": "John", "age": 30}
```

### With Enum Parser

```python
from langchain_classic.output_parsers.enum import EnumOutputParser
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

base_parser = EnumOutputParser(enum=Sentiment)
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI()
)

# LLM might return "very positive" instead of "positive"
result = fixing_parser.parse("very positive")
# Fixing parser asks LLM to correct it to one of the valid values
# Returns: Sentiment.POSITIVE
```

### Multiple Retries

```python
from langchain_classic.output_parsers.datetime import DatetimeOutputParser

base_parser = DatetimeOutputParser()
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(),
    max_retries=3  # Try up to 3 times
)

# Invalid datetime format
result = fixing_parser.parse("July 4th, 2023 at 2:30 PM")
# Will retry up to 3 times to get correct format
# Returns: datetime.datetime(2023, 7, 4, 14, 30, 0, 0)
```

### Custom Fix Prompt

```python
from langchain_core.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    template="""The following output was rejected by the parser:
{completion}

Error: {error}

Please fix the output to match these instructions:
{instructions}

Return ONLY the fixed output, nothing else.""",
    input_variables=["completion", "error", "instructions"]
)

fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(),
    prompt=custom_prompt
)
```

### Async Usage

```python
import asyncio

async def parse_with_fixing():
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=ChatOpenAI()
    )

    result = await fixing_parser.aparse("invalid output")
    return result

result = asyncio.run(parse_with_fixing())
```

### In a Chain

```python
from langchain_core.prompts import PromptTemplate

# Create chain with fixing parser
prompt = PromptTemplate(
    template="Extract person info from: {text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": fixing_parser.get_format_instructions()}
)

chain = prompt | ChatOpenAI() | fixing_parser

result = chain.invoke({"text": "John is 30 years old"})
# Returns parsed and validated structure, with automatic error correction
```

### Error Handling

```python
from langchain_core.exceptions import OutputParserException

fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(),
    max_retries=2
)

try:
    # Even fixing parser can fail after max retries
    result = fixing_parser.parse("completely unparseable garbage")
except OutputParserException as e:
    print(f"Failed to parse after {fixing_parser.max_retries} retries: {e}")
```

### Monitoring Retries

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Set up parser with logging
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(),
    max_retries=3
)

# You can track retry attempts via logs
result = fixing_parser.parse("bad input")
```

## Related Pages

- `BaseOutputParser` - Base class for all output parsers
- `RetryOutputParser` - Alternative retry parser with different strategy
- `RetryWithErrorOutputParser` - Retry parser that includes original input
- `StructuredOutputParser` - Common parser to wrap for JSON validation
- `EnumOutputParser` - Common parser to wrap for enum validation
- `NAIVE_FIX_PROMPT` - Default prompt template in `output_parsers/prompts.py`

## Implementation Notes

### Design Decisions

1. **Wrapper Pattern**: Wraps any parser without modifying it, following composition over inheritance
2. **Generic Type**: Preserves type safety of wrapped parser using TypeVar
3. **Incremental Retry**: Updates completion iteratively rather than starting fresh
4. **Error Context**: Provides full context (instructions, completion, error) to fixing LLM
5. **Graceful Fallback**: Handles parsers without `get_format_instructions()` method
6. **Dual API Support**: Supports both legacy (run/arun) and modern (invoke/ainvoke) APIs

### Retry Strategy

The retry mechanism:
1. Attempts parse with base parser
2. On failure, invokes LLM with:
   - Original completion
   - Parser's format instructions
   - Error message from failed parse
3. LLM generates corrected completion
4. Replaces original completion with fixed version
5. Retries parse with fixed completion
6. Repeats until success or max_retries exceeded

This incremental approach preserves as much of the original content as possible while fixing format issues.

### Cost Considerations

Each retry makes an additional LLM call:
- **Cost**: Retries multiply API costs
- **Latency**: Each retry adds ~1-3 seconds
- **Trade-off**: Balance robustness vs. cost by tuning `max_retries`

**Recommendations:**
- Use `max_retries=1` for most cases
- Increase to 2-3 for critical parsing
- Consider implementing retry limits at application level

### Performance Considerations

- **Success Case**: No overhead if base parser succeeds first time
- **Failure Case**: Each retry adds full LLM inference time
- **Async Support**: Fully async-capable for concurrent operations
- **No Caching**: Each retry is independent (no memoization)

### Limitations

1. **No Guarantee**: Even with retries, parsing can still fail
2. **Cost Multiplier**: Failed parses incur multiple LLM costs
3. **Latency Impact**: Retries add significant latency
4. **Context Length**: Long completions + errors may exceed LLM context limits
5. **No Learning**: Each parse is independent; doesn't learn from previous failures
6. **Determinism**: Non-deterministic due to LLM fixing (same input may yield different results)

### Best Practices

1. **Start Conservative**: Use `max_retries=1` and increase if needed
2. **Monitor Failures**: Track retry rates to identify systematic issues
3. **Improve Prompts**: Better initial prompts reduce retry needs
4. **Use Appropriate LLM**: Fixing LLM should be capable enough to understand errors
5. **Validate Results**: Even fixed outputs should be validated downstream
6. **Consider Alternatives**: For high-volume parsing, fix prompt issues upstream
7. **Set Timeouts**: Implement timeouts to prevent excessive retry chains

### Use Cases

- JSON parsing with occasional formatting errors
- Enum selection with fuzzy LLM outputs
- Datetime parsing with varied formats
- Structured data extraction with inconsistent LLM formatting
- Production systems requiring high reliability
- User-facing applications where failures are unacceptable
- Complex schemas where perfect prompting is difficult

### When NOT to Use

- High-volume parsing (cost prohibitive)
- Low-latency requirements (retries add delay)
- Deterministic requirements (fixing is non-deterministic)
- When base parser always succeeds
- Simple parsers (e.g., string extraction)
- When failures indicate deeper issues (fix root cause instead)

### Integration Patterns

**Pattern 1: Graceful Degradation**
```python
try:
    result = base_parser.parse(text)
except OutputParserException:
    result = fixing_parser.parse(text)
```

**Pattern 2: Monitoring Wrapper**
```python
class MonitoredFixingParser(OutputFixingParser):
    def parse(self, completion: str):
        try:
            return self.parser.parse(completion)
        except OutputParserException:
            metrics.increment("parser.retries")
            return super().parse(completion)
```

**Pattern 3: Fallback Chain**
```python
# Try multiple strategies
try:
    return simple_parser.parse(text)
except:
    try:
        return fixing_parser.parse(text)
    except:
        return fallback_parser.parse(text)
```
