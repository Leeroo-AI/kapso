{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|Python Exceptions|https://docs.python.org/3/tutorial/errors.html]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Structured_Output]], [[domain::Error_Handling]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for handling and recovering from structured output parsing failures, provided by LangChain's agent system.

=== Description ===

Structured output error handling consists of:

1. **Error Classes** (`structured_output.py`):
   - `StructuredOutputError`: Base class with `ai_message` reference
   - `MultipleStructuredOutputsError`: When model returns multiple output tool calls
   - `StructuredOutputValidationError`: When parsing fails against schema

2. **Error Handler** (`factory.py`):
   - `_handle_structured_output_error`: Decides retry vs propagate based on `handle_errors` config

These components enable resilient structured output extraction with configurable retry behavior.

=== Usage ===

Use error handling (automatically via ToolStrategy) when:
* Schema validation may fail due to LLM inconsistency
* Building robust extraction pipelines
* Want automatic retry with feedback

Configure via `ToolStrategy.handle_errors` parameter.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/structured_output.py
* '''Lines:''' L34-73 (error classes)
* '''File:''' libs/langchain_v1/langchain/agents/factory.py
* '''Lines:''' L401-428 (_handle_structured_output_error)

=== Signature ===
<syntaxhighlight lang="python">
class StructuredOutputError(Exception):
    """Base class for structured output errors."""
    ai_message: AIMessage


class MultipleStructuredOutputsError(StructuredOutputError):
    """Raised when model returns multiple structured output tool calls
    when only one is expected."""

    def __init__(self, tool_names: list[str], ai_message: AIMessage) -> None:
        """Initialize MultipleStructuredOutputsError.

        Args:
            tool_names: The names of the tools called for structured output.
            ai_message: The AI message that contained the invalid multiple tool calls.
        """


class StructuredOutputValidationError(StructuredOutputError):
    """Raised when structured output tool call arguments fail to parse
    according to the schema."""

    def __init__(self, tool_name: str, source: Exception, ai_message: AIMessage) -> None:
        """Initialize StructuredOutputValidationError.

        Args:
            tool_name: The name of the tool that failed.
            source: The exception that occurred.
            ai_message: The AI message that contained the invalid structured output.
        """


def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat,
) -> tuple[bool, str]:
    """Handle structured output error.

    Returns:
        (should_retry, retry_tool_message) - whether to retry and feedback message
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import (
    StructuredOutputError,
    MultipleStructuredOutputsError,
    StructuredOutputValidationError,
)

# Error handler is internal, used automatically by agent
</syntaxhighlight>

== I/O Contract ==

=== Inputs (_handle_structured_output_error) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| exception || Exception || Yes || The exception from parsing
|-
| response_format || ResponseFormat || Yes || Strategy with handle_errors config
|}

=== Outputs (_handle_structured_output_error) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| should_retry || bool || Whether agent should retry with feedback
|-
| retry_message || str || Error message for LLM feedback (empty if no retry)
|}

== Usage Examples ==

=== Default Error Handling (Retry All) ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class StrictOutput(BaseModel):
    """Output with strict validation."""
    name: str = Field(min_length=1)
    value: int = Field(ge=0, le=100)


# Default: handle_errors=True
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    response_format=ToolStrategy(StrictOutput),  # handle_errors=True by default
)

# If model returns {"name": "", "value": 150} (invalid):
# 1. Parsing fails (name too short, value > 100)
# 2. Agent retries with error feedback
# 3. Model tries again with corrected output
result = agent.invoke({"messages": [{"role": "user", "content": "Generate output"}]})
</syntaxhighlight>

=== Disabled Error Handling (Propagate) ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, StructuredOutputValidationError


class Output(BaseModel):
    value: int


# Disable retry - errors propagate
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    response_format=ToolStrategy(Output, handle_errors=False),
)

try:
    result = agent.invoke({"messages": [msg]})
except StructuredOutputValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Tool: {e.tool_name}")
    print(f"Source: {e.source}")
    print(f"Original message: {e.ai_message}")
</syntaxhighlight>

=== Custom Error Message ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class JSONOutput(BaseModel):
    data: dict


# Custom message for retry
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    response_format=ToolStrategy(
        JSONOutput,
        handle_errors="Output must be valid JSON with a 'data' object. Please fix and try again."
    ),
)

# On error, model receives custom feedback message
</syntaxhighlight>

=== Selective Exception Handling ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import ValidationError


class TypedOutput(BaseModel):
    count: int
    name: str


# Only retry on ValidationError, propagate others
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    response_format=ToolStrategy(
        TypedOutput,
        handle_errors=ValidationError,  # Only catch this type
    ),
)

# ValidationError (e.g., wrong type): retries
# Other errors (e.g., JSON parse error): propagates
</syntaxhighlight>

=== Multiple Exception Types ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import ValidationError
from json import JSONDecodeError


class Output(BaseModel):
    result: str


# Retry on multiple exception types
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    response_format=ToolStrategy(
        Output,
        handle_errors=(ValidationError, KeyError, TypeError),
    ),
)
</syntaxhighlight>

=== Custom Error Handler Function ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class APIResponse(BaseModel):
    status: str
    data: dict


def smart_error_handler(exception: Exception) -> str:
    """Generate context-aware error feedback."""
    error_str = str(exception).lower()

    if "required" in error_str:
        return "Missing required field. Include 'status' (string) and 'data' (object)."
    if "type" in error_str:
        return "Type error. Ensure 'status' is a string and 'data' is an object/dict."
    if "json" in error_str:
        return "Invalid JSON. Check for proper quotes, brackets, and commas."

    return f"Validation failed: {exception}. Please output valid JSON matching the schema."


agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[],
    response_format=ToolStrategy(APIResponse, handle_errors=smart_error_handler),
)
</syntaxhighlight>

=== Catching Structured Output Errors ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import (
    StructuredOutputError,
    MultipleStructuredOutputsError,
    StructuredOutputValidationError,
)

try:
    result = agent.invoke({"messages": [msg]})
except MultipleStructuredOutputsError as e:
    print(f"Model returned multiple outputs: {e.tool_names}")
    print(f"Expected single output, got {len(e.tool_names)}")
except StructuredOutputValidationError as e:
    print(f"Validation failed for tool '{e.tool_name}'")
    print(f"Original error: {e.source}")
except StructuredOutputError as e:
    print(f"Generic structured output error")
    print(f"AI message: {e.ai_message}")
</syntaxhighlight>

=== Inspecting Error Details ===
<syntaxhighlight lang="python">
from langchain.agents.structured_output import StructuredOutputValidationError

try:
    result = agent.invoke({"messages": [msg]})
except StructuredOutputValidationError as e:
    # Access error details
    print(f"Tool name: {e.tool_name}")
    print(f"Source exception: {e.source}")
    print(f"Source type: {type(e.source)}")

    # Access original AI response
    ai_msg = e.ai_message
    print(f"Tool calls: {ai_msg.tool_calls}")

    # Can manually retry or log for debugging
    if ai_msg.tool_calls:
        raw_args = ai_msg.tool_calls[0]["args"]
        print(f"Raw args that failed: {raw_args}")
</syntaxhighlight>

=== Error Handling in Agent Factory ===
<syntaxhighlight lang="python">
# Internal usage in agent factory (simplified)
from langchain.agents.structured_output import ToolStrategy

def process_response(
    response: AIMessage,
    binding: OutputToolBinding,
    response_format: ResponseFormat
) -> dict:
    """Process model response with error handling."""
    try:
        result = binding.parse(response.tool_calls[0]["args"])
        return {"structured_response": result}

    except Exception as e:
        # Check if should retry
        should_retry, msg = _handle_structured_output_error(e, response_format)

        if should_retry:
            # Return feedback for agent loop to continue
            return {
                "messages": [
                    response,
                    ToolMessage(content=msg, tool_call_id=response.tool_calls[0]["id"])
                ]
            }
        else:
            # Wrap and raise
            raise StructuredOutputValidationError(
                tool_name=binding.tool.name,
                source=e,
                ai_message=response
            )
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Structured_Output_Error_Handling]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
