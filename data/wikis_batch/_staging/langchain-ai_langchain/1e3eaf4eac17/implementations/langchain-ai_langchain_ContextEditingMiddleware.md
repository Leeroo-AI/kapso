# ContextEditingMiddleware Implementation

## Metadata
- **Component**: `ContextEditingMiddleware`
- **Package**: `langchain.agents.middleware.context_editing`
- **File Path**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/context_editing.py`
- **Type**: Agent Middleware
- **Lines of Code**: 278
- **Related Classes**: `ClearToolUsesEdit`, `ContextEdit` (Protocol)

## Overview

`ContextEditingMiddleware` is a sophisticated agent middleware component that automatically manages context window size by pruning tool results when the conversation exceeds configurable token thresholds. This implementation mirrors Anthropic's context editing capabilities (specifically the `clear_tool_uses_20250919` behavior) while remaining model-agnostic to work with any LangChain chat model.

### Purpose
The middleware addresses the challenge of managing long-running agent conversations by strategically clearing older tool results to prevent context window overflow. This ensures agents can continue operating effectively in extended sessions without hitting token limits.

### Key Features
- **Token-based triggering**: Monitors conversation token count and activates when thresholds are exceeded
- **Configurable preservation**: Keeps the N most recent tool results intact
- **Selective exclusion**: Allows specific tools to be excluded from clearing
- **Smart clearing**: Optionally clears both tool outputs and inputs
- **Token reclamation**: Can specify minimum tokens to reclaim per edit operation
- **Model-agnostic**: Works with both approximate and exact token counting methods

## Code Reference

### Main Class Definition

```python
class ContextEditingMiddleware(AgentMiddleware):
    """Automatically prune tool results to manage context size.

    The middleware applies a sequence of edits when the total input token count exceeds
    configured thresholds.

    Currently the `ClearToolUsesEdit` strategy is supported, aligning with Anthropic's
    `clear_tool_uses_20250919` behavior.
    """

    edits: list[ContextEdit]
    token_count_method: Literal["approximate", "model"]

    def __init__(
        self,
        *,
        edits: Iterable[ContextEdit] | None = None,
        token_count_method: Literal["approximate", "model"] = "approximate",
    ) -> None:
        """Initialize an instance of context editing middleware.

        Args:
            edits: Sequence of edit strategies to apply.
                Defaults to a single `ClearToolUsesEdit` mirroring Anthropic defaults.
            token_count_method: Whether to use approximate token counting
                (faster, less accurate) or exact counting implemented by the
                chat model (potentially slower, more accurate).
        """
```

### ClearToolUsesEdit Strategy

```python
@dataclass(slots=True)
class ClearToolUsesEdit(ContextEdit):
    """Configuration for clearing tool outputs when token limits are exceeded."""

    trigger: int = 100_000
    """Token count that triggers the edit."""

    clear_at_least: int = 0
    """Minimum number of tokens to reclaim when the edit runs."""

    keep: int = 3
    """Number of most recent tool results that must be preserved."""

    clear_tool_inputs: bool = False
    """Whether to clear the originating tool call parameters on the AI message."""

    exclude_tools: Sequence[str] = ()
    """List of tool names to exclude from clearing."""

    placeholder: str = DEFAULT_TOOL_PLACEHOLDER
    """Placeholder text inserted for cleared tool outputs."""
```

### Core Logic

The `apply` method in `ClearToolUsesEdit` implements the clearing strategy:

```python
def apply(
    self,
    messages: list[AnyMessage],
    *,
    count_tokens: TokenCounter,
) -> None:
    """Apply the clear-tool-uses strategy."""
    tokens = count_tokens(messages)

    if tokens <= self.trigger:
        return

    candidates = [
        (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
    ]

    if self.keep >= len(candidates):
        candidates = []
    elif self.keep:
        candidates = candidates[: -self.keep]

    cleared_tokens = 0
    excluded_tools = set(self.exclude_tools)

    for idx, tool_message in candidates:
        if tool_message.response_metadata.get("context_editing", {}).get("cleared"):
            continue

        # Find corresponding AI message and tool call
        ai_message = next(
            (m for m in reversed(messages[:idx]) if isinstance(m, AIMessage)), None
        )

        if ai_message is None:
            continue

        tool_call = next(
            (
                call
                for call in ai_message.tool_calls
                if call.get("id") == tool_message.tool_call_id
            ),
            None,
        )

        if tool_call is None:
            continue

        if (tool_message.name or tool_call["name"]) in excluded_tools:
            continue

        # Replace tool message with cleared version
        messages[idx] = tool_message.model_copy(
            update={
                "artifact": None,
                "content": self.placeholder,
                "response_metadata": {
                    **tool_message.response_metadata,
                    "context_editing": {
                        "cleared": True,
                        "strategy": "clear_tool_uses",
                    },
                },
            }
        )

        if self.clear_tool_inputs:
            messages[messages.index(ai_message)] = self._build_cleared_tool_input_message(
                ai_message,
                tool_message.tool_call_id,
            )

        if self.clear_at_least > 0:
            new_token_count = count_tokens(messages)
            cleared_tokens = max(0, tokens - new_token_count)
            if cleared_tokens >= self.clear_at_least:
                break
```

## I/O Contract

### Input
- **ModelRequest**: Contains the conversation messages and model configuration
- **Configuration Parameters**:
  - `edits`: List of `ContextEdit` strategies to apply (defaults to single `ClearToolUsesEdit`)
  - `token_count_method`: Either `"approximate"` (fast) or `"model"` (accurate)

### Output
- **ModelResponse**: Modified request with edited message history passed to next handler
- **Side Effects**: Messages are modified in place with cleared tool results marked by metadata

### State Modifications
The middleware modifies `ToolMessage` objects by:
1. Replacing `content` with placeholder text (default: `"[cleared]"`)
2. Setting `artifact` to `None`
3. Adding `response_metadata.context_editing.cleared = True`
4. Optionally clearing tool call arguments in corresponding `AIMessage`

## Usage Examples

### Basic Usage with Defaults

```python
from langchain.agents.middleware.context_editing import ContextEditingMiddleware
from langchain.agents import create_agent

# Create middleware with default settings (100k token trigger)
middleware = ContextEditingMiddleware()

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[middleware],
)

# Agent will automatically clear tool results when context exceeds 100k tokens
result = await agent.invoke({"messages": [HumanMessage("Long running task")]})
```

### Custom Configuration

```python
from langchain.agents.middleware.context_editing import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
)

# Create custom edit strategy
custom_edit = ClearToolUsesEdit(
    trigger=50_000,              # Trigger at 50k tokens instead of 100k
    clear_at_least=10_000,       # Reclaim at least 10k tokens per operation
    keep=5,                      # Keep 5 most recent tool results
    clear_tool_inputs=True,      # Also clear tool call arguments
    exclude_tools=["critical_tool"],  # Never clear this tool
    placeholder="[Tool output removed to save context]",
)

middleware = ContextEditingMiddleware(
    edits=[custom_edit],
    token_count_method="model",  # Use exact model token counting
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[search_tool, calculator_tool, critical_tool],
    middleware=[middleware],
)
```

### Multiple Edit Strategies

```python
# Apply multiple edit strategies in sequence
edits = [
    ClearToolUsesEdit(
        trigger=80_000,
        keep=5,
        exclude_tools=["critical_tool"],
    ),
    ClearToolUsesEdit(
        trigger=100_000,
        keep=3,
        clear_tool_inputs=True,
    ),
]

middleware = ContextEditingMiddleware(edits=edits)
```

### Token Counting Methods

```python
# Fast approximate counting (default)
fast_middleware = ContextEditingMiddleware(
    token_count_method="approximate"
)

# Accurate model-based counting (slower but precise)
accurate_middleware = ContextEditingMiddleware(
    token_count_method="model"
)
```

## Implementation Details

### Token Counting Strategy

The middleware supports two token counting methods:

1. **Approximate** (default): Uses `count_tokens_approximately` from `langchain_core.messages.utils`
   - Faster execution
   - Less accurate counts
   - Good for most use cases

2. **Model**: Uses the chat model's `get_num_tokens_from_messages` method
   - More accurate counts
   - Potentially slower
   - Includes system messages and tools in count

### Message Modification Process

1. **Count Tokens**: Calculate current conversation token count
2. **Check Trigger**: If count exceeds trigger threshold, proceed to clearing
3. **Identify Candidates**: Find all `ToolMessage` objects excluding the N most recent
4. **Filter Excluded**: Skip tool messages for excluded tool names
5. **Clear Messages**: Replace content with placeholder and mark as cleared
6. **Optional Input Clearing**: If enabled, clear tool call arguments in AI messages
7. **Token Verification**: If `clear_at_least` set, verify minimum tokens reclaimed

### Metadata Tracking

Cleared messages are marked with metadata to prevent double-clearing:

```python
response_metadata = {
    "context_editing": {
        "cleared": True,
        "strategy": "clear_tool_uses",
    }
}
```

For cleared tool inputs:

```python
response_metadata = {
    "context_editing": {
        "cleared_tool_inputs": ["tool_call_id_1", "tool_call_id_2"],
    }
}
```

### Deep Copy Protection

The middleware creates a deep copy of messages before modification to prevent side effects:

```python
edited_messages = deepcopy(list(request.messages))
for edit in self.edits:
    edit.apply(edited_messages, count_tokens=count_tokens)
```

## Related Pages

### Core Middleware Infrastructure
- **langchain-ai_langchain_AgentMiddleware_class.md**: Base middleware interface
- **langchain-ai_langchain_middleware_hooks.txt**: Middleware hook system

### Other V1 Middleware Implementations
- **langchain-ai_langchain_ModelCallLimitMiddleware.md**: Model call quota enforcement
- **langchain-ai_langchain_ModelFallbackMiddleware.md**: Model failover on errors
- **langchain-ai_langchain_TodoListMiddleware.md**: Task tracking for agents
- **langchain-ai_langchain_LLMToolEmulator.md**: LLM-based tool emulation

### Related Middleware
- **langchain-ai_langchain_ModelRetryMiddleware.md**: Retry logic for model calls
- **langchain-ai_langchain_ToolRetryMiddleware.md**: Retry logic for tool calls

### Agent Creation
- **langchain-ai_langchain_create_agent.md**: Agent creation with middleware support

## Architecture Notes

### Design Philosophy
The middleware follows these principles:
- **Model-agnostic**: Works with any LangChain chat model
- **Non-destructive**: Uses metadata to track cleared state
- **Configurable**: Multiple customization points for different use cases
- **Transparent**: Middleware operation is visible through metadata

### Performance Considerations
- Approximate token counting is recommended for most use cases
- Deep copying messages has memory overhead
- Token counting happens before every model call
- Clearing operations are O(n) where n is the number of messages

### Anthropic Alignment
This implementation mirrors Anthropic's `clear_tool_uses_20250919` behavior:
- 100k token default trigger
- Keeps recent tool results
- Clears older outputs first
- Preserves message structure

## Extension Points

### Custom Edit Strategies

Implement the `ContextEdit` protocol to create custom strategies:

```python
class ContextEdit(Protocol):
    """Protocol describing a context editing strategy."""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply an edit to the message list in place."""
        ...
```

### Custom Token Counters

The `TokenCounter` type allows custom counting logic:

```python
TokenCounter = Callable[
    [Sequence[BaseMessage]],
    int,
]
```
