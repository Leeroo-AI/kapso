{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Structured Output|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|Retry Pattern|https://docs.microsoft.com/en-us/azure/architecture/patterns/retry]]
* [[source::Doc|Python Exceptions|https://docs.python.org/3/tutorial/errors.html]]
|-
! Domains
| [[domain::LLM]], [[domain::Structured_Output]], [[domain::Error_Handling]], [[domain::Resilience]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Strategy for handling parsing failures in structured output extraction, including error classification, retry decision, and feedback generation.

=== Description ===

Structured Output Error Handling provides resilience when LLM responses don't match expected schemas. The system:

* Classifies errors (validation errors, multiple outputs, JSON parse failures)
* Decides whether to retry based on `handle_errors` configuration
* Generates feedback messages for retry attempts
* Propagates unrecoverable errors

This handling is primarily used with `ToolStrategy`, which supports configurable error handling. `ProviderStrategy` relies on provider-side enforcement and typically doesn't retry.

=== Usage ===

Error handling activates when:
* Tool call arguments fail schema validation
* Model returns multiple structured output calls when only one expected
* JSON parsing fails (ProviderStrategy)

The `handle_errors` parameter in `ToolStrategy` controls behavior:
* `True`: Retry with default error message
* `False`: Propagate exception
* `str`: Retry with custom message
* `type[Exception]`: Retry only for specific exception types
* `Callable`: Custom error handler function

== Theoretical Basis ==

Structured Output Error Handling implements **Retry Pattern** with **Configurable Error Policy**.

'''1. Error Classification'''

<syntaxhighlight lang="python">
# Custom error classes for structured output
class StructuredOutputError(Exception):
    """Base class for structured output errors."""
    ai_message: AIMessage  # Original message that caused error


class MultipleStructuredOutputsError(StructuredOutputError):
    """Raised when model returns multiple structured output tool calls
    when only one is expected."""
    tool_names: list[str]  # Names of tools called


class StructuredOutputValidationError(StructuredOutputError):
    """Raised when structured output tool call arguments fail to parse
    according to the schema."""
    tool_name: str        # Tool that failed
    source: Exception     # Original validation exception
</syntaxhighlight>

'''2. Error Handling Configuration'''

<syntaxhighlight lang="python">
from langchain.agents.structured_output import ToolStrategy

# Option 1: Retry on all errors (default)
strategy = ToolStrategy(Schema, handle_errors=True)

# Option 2: No retry, propagate errors
strategy = ToolStrategy(Schema, handle_errors=False)

# Option 3: Custom error message for retry
strategy = ToolStrategy(
    Schema,
    handle_errors="Please output valid JSON matching the schema exactly."
)

# Option 4: Only retry on specific exceptions
from pydantic import ValidationError
strategy = ToolStrategy(Schema, handle_errors=ValidationError)

# Option 5: Retry on multiple exception types
strategy = ToolStrategy(Schema, handle_errors=(ValidationError, KeyError))

# Option 6: Custom error handler function
def custom_handler(exception: Exception) -> str:
    return f"Error occurred: {exception}. Please fix and try again."

strategy = ToolStrategy(Schema, handle_errors=custom_handler)
</syntaxhighlight>

'''3. Error Handling Decision Flow'''

<syntaxhighlight lang="text">
Parsing Exception                  Decision
┌─────────────────┐              ┌─────────────────┐
│ ValidationError │              │ Check           │
│ from parsing    │─────────────►│ handle_errors   │
└─────────────────┘              │ configuration   │
                                 └────────┬────────┘
                                          │
           ┌──────────────────────────────┼──────────────────────────────┐
           │                              │                              │
           ▼                              ▼                              ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ handle_errors=   │          │ handle_errors=   │          │ handle_errors=   │
│ False            │          │ True / str       │          │ type/callable    │
│                  │          │                  │          │                  │
│ Return:          │          │ Return:          │          │ Check if         │
│ (False, "")      │          │ (True, message)  │          │ matches type     │
│ → Propagate      │          │ → Retry          │          │ → Retry/Propagate│
└──────────────────┘          └──────────────────┘          └──────────────────┘
</syntaxhighlight>

'''4. _handle_structured_output_error Function'''

<syntaxhighlight lang="python">
STRUCTURED_OUTPUT_ERROR_TEMPLATE = """Your previous response didn't match the expected schema.
Error: {error}
Please try again with valid output matching the schema."""

def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat,
) -> tuple[bool, str]:
    """Handle structured output error.

    Returns:
        (should_retry, retry_tool_message)
    """
    # Only ToolStrategy supports error handling
    if not isinstance(response_format, ToolStrategy):
        return False, ""

    handle_errors = response_format.handle_errors

    # Boolean handling
    if handle_errors is False:
        return False, ""
    if handle_errors is True:
        return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))

    # String: custom message
    if isinstance(handle_errors, str):
        return True, handle_errors

    # Exception type: check if matches
    if isinstance(handle_errors, type) and issubclass(handle_errors, Exception):
        if isinstance(exception, handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""

    # Tuple of exception types
    if isinstance(handle_errors, tuple):
        if any(isinstance(exception, exc_type) for exc_type in handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""

    # Callable: custom handler
    if callable(handle_errors):
        return True, handle_errors(exception)

    return False, ""
</syntaxhighlight>

'''5. Retry Flow in Agent Loop'''

<syntaxhighlight lang="python">
# Simplified agent loop with error handling
def process_structured_output(response, output_binding, response_format):
    try:
        # Attempt to parse
        result = output_binding.parse(response.tool_calls[0]["args"])
        return {"structured_response": result}

    except Exception as e:
        # Wrap in structured output error
        error = StructuredOutputValidationError(
            tool_name=output_binding.tool.name,
            source=e,
            ai_message=response
        )

        # Check if should retry
        should_retry, error_message = _handle_structured_output_error(
            error, response_format
        )

        if should_retry:
            # Return error feedback for retry
            return {
                "messages": [
                    response,  # Original AI message
                    ToolMessage(  # Error feedback
                        content=error_message,
                        tool_call_id=response.tool_calls[0]["id"]
                    )
                ]
            }
        else:
            # Propagate error
            raise error
</syntaxhighlight>

'''6. Error Messages for LLM Feedback'''

<syntaxhighlight lang="python">
# Default error template provides context for retry
STRUCTURED_OUTPUT_ERROR_TEMPLATE = """Your previous response didn't match the expected schema.
Error: {error}
Please try again with valid output matching the schema."""

# Custom messages can be more specific
custom_message = """Invalid JSON structure. Please ensure:
1. All required fields are present
2. Data types match the schema
3. No extra fields are included
"""

# Callable for dynamic messages
def dynamic_handler(e: Exception) -> str:
    if "required" in str(e).lower():
        return "Missing required field. Check the schema and include all required fields."
    if "type" in str(e).lower():
        return "Type mismatch. Ensure numeric fields are numbers, strings are quoted, etc."
    return f"Validation error: {e}"
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_structured_output_error_classes]]

=== Used By Workflows ===
* Structured_Output_Workflow (Step 6)
