# ModelFallbackMiddleware Implementation

## Metadata
- **Component**: `ModelFallbackMiddleware`
- **Package**: `langchain.agents.middleware.model_fallback`
- **File Path**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/model_fallback.py`
- **Type**: Agent Middleware
- **Lines of Code**: 135
- **Related Classes**: None

## Overview

`ModelFallbackMiddleware` provides automatic failover to alternative models when the primary model encounters errors. It implements a sequential fallback strategy that retries failed model calls with alternative models in order until success or all models are exhausted.

### Purpose
The middleware addresses reliability concerns in production agent systems by providing graceful degradation when models fail due to rate limits, service outages, or other errors. This ensures agents can continue operating even when specific model providers experience issues.

### Key Features
- **Sequential fallback**: Try fallback models in specified order
- **Transparent operation**: Maintains same interface as direct model calls
- **Flexible configuration**: Supports model strings or pre-initialized instances
- **Exception propagation**: Re-raises last exception if all models fail
- **Model override**: Seamlessly substitutes models in request

## Code Reference

### Main Class Definition

```python
class ModelFallbackMiddleware(AgentMiddleware):
    """Automatic fallback to alternative models on errors.

    Retries failed model calls with alternative models in sequence until
    success or all models exhausted. Primary model specified in `create_agent`.

    Example:
        ```python
        from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
        from langchain.agents import create_agent

        fallback = ModelFallbackMiddleware(
            "openai:gpt-4o-mini",  # Try first on error
            "anthropic:claude-sonnet-4-5-20250929",  # Then this
        )

        agent = create_agent(
            model="openai:gpt-4o",  # Primary model
            middleware=[fallback],
        )

        # If primary fails: tries gpt-4o-mini, then claude-sonnet-4-5-20250929
        result = await agent.invoke({"messages": [HumanMessage("Hello")]})
        ```
    """

    def __init__(
        self,
        first_model: str | BaseChatModel,
        *additional_models: str | BaseChatModel,
    ) -> None:
        """Initialize model fallback middleware.

        Args:
            first_model: First fallback model (string name or instance).
            *additional_models: Additional fallbacks in order.
        """
        super().__init__()

        # Initialize all fallback models
        all_models = (first_model, *additional_models)
        self.models: list[BaseChatModel] = []
        for model in all_models:
            if isinstance(model, str):
                self.models.append(init_chat_model(model))
            else:
                self.models.append(model)
```

### Sync Fallback Logic

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelCallResult:
    """Try fallback models in sequence on errors.

    Args:
        request: Initial model request.
        handler: Callback to execute the model.

    Returns:
        AIMessage from successful model call.

    Raises:
        Exception: If all models fail, re-raises last exception.
    """
    # Try primary model first
    last_exception: Exception
    try:
        return handler(request)
    except Exception as e:
        last_exception = e

    # Try fallback models
    for fallback_model in self.models:
        try:
            return handler(request.override(model=fallback_model))
        except Exception as e:
            last_exception = e
            continue

    raise last_exception
```

### Async Fallback Logic

```python
async def awrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelCallResult:
    """Try fallback models in sequence on errors (async version).

    Args:
        request: Initial model request.
        handler: Async callback to execute the model.

    Returns:
        AIMessage from successful model call.

    Raises:
        Exception: If all models fail, re-raises last exception.
    """
    # Try primary model first
    last_exception: Exception
    try:
        return await handler(request)
    except Exception as e:
        last_exception = e

    # Try fallback models
    for fallback_model in self.models:
        try:
            return await handler(request.override(model=fallback_model))
        except Exception as e:
            last_exception = e
            continue

    raise last_exception
```

## I/O Contract

### Input
- **ModelRequest**: Contains the conversation messages, system prompt, tools, and primary model
- **Fallback Models**: List of alternative models (strings or instances)

### Output
- **ModelResponse**: AIMessage from the first successful model call
- **Exception**: Last exception raised if all models (primary + fallbacks) fail

### Execution Flow
1. Handler invokes primary model (specified in `create_agent`)
2. On exception, try first fallback model
3. On exception, try second fallback model
4. Continue until success or all models exhausted
5. If all fail, re-raise last exception

### Model Override Mechanism
The middleware uses `request.override(model=fallback_model)` to substitute models while preserving:
- Messages
- System prompt
- Tools
- Other request parameters

## Usage Examples

### Basic Fallback Chain

```python
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
from langchain.agents import create_agent

# Create fallback chain: gpt-4o -> gpt-4o-mini -> claude
fallback = ModelFallbackMiddleware(
    "openai:gpt-4o-mini",
    "anthropic:claude-sonnet-4-5-20250929",
)

agent = create_agent(
    model="openai:gpt-4o",  # Primary model
    tools=[search_tool, calculator_tool],
    middleware=[fallback],
)

# If gpt-4o fails -> tries gpt-4o-mini
# If gpt-4o-mini fails -> tries claude
# If claude fails -> raises exception
result = await agent.invoke({"messages": [HumanMessage("Calculate 2+2")]})
```

### Cross-Provider Resilience

```python
# Fallback across different providers for maximum reliability
fallback = ModelFallbackMiddleware(
    "anthropic:claude-sonnet-4-5-20250929",  # Different provider
    "google:gemini-2-flash-exp",              # Another provider
    "openai:gpt-4o-mini",                     # Back to OpenAI cheaper model
)

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[fallback],
)

# Resilient against provider-specific outages
```

### Pre-Initialized Models

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Use pre-configured model instances
primary_model = ChatOpenAI(model="gpt-4o", temperature=0.7)
fallback_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)

fallback = ModelFallbackMiddleware(fallback_model)

agent = create_agent(
    model=primary_model,
    middleware=[fallback],
)
```

### Cost-Aware Fallback

```python
# Start with expensive model, fallback to cheaper alternatives
fallback = ModelFallbackMiddleware(
    "openai:gpt-4o-mini",      # 15x cheaper than gpt-4o
    "anthropic:claude-haiku",  # Even cheaper
)

agent = create_agent(
    model="openai:gpt-4o",     # Most capable but expensive
    middleware=[fallback],
)

# Try expensive model first, degrade to cheaper on failure
```

### Rate Limit Handling

```python
# Multiple instances of same model for rate limit recovery
fallback = ModelFallbackMiddleware(
    "openai:gpt-4o",  # Try again (might work if transient rate limit)
    "openai:gpt-4o",  # One more try
    "openai:gpt-4o-mini",  # Finally, cheaper model
)

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[fallback],
)

# Gives multiple chances for rate limit errors to clear
```

### Error Handling

```python
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware

fallback = ModelFallbackMiddleware(
    "openai:gpt-4o-mini",
    "anthropic:claude-sonnet-4-5-20250929",
)

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[fallback],
)

try:
    result = await agent.invoke({"messages": [HumanMessage("Hello")]})
except Exception as e:
    # All models failed - last exception is raised
    print(f"All models exhausted: {e}")
    # Could log to monitoring, retry with different strategy, etc.
```

### Single Fallback

```python
# Simple primary -> backup pattern
fallback = ModelFallbackMiddleware("openai:gpt-4o-mini")

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[fallback],
)

# Just one fallback - gpt-4o -> gpt-4o-mini
```

### Mixing String and Instance

```python
from langchain_anthropic import ChatAnthropic

# Mix string identifiers and model instances
custom_claude = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0.9,
    max_tokens=4000,
)

fallback = ModelFallbackMiddleware(
    "openai:gpt-4o-mini",  # String identifier
    custom_claude,          # Pre-configured instance
)
```

## Implementation Details

### Model Initialization

The middleware converts all model specifications to `BaseChatModel` instances during initialization:

```python
for model in all_models:
    if isinstance(model, str):
        self.models.append(init_chat_model(model))
    else:
        self.models.append(model)
```

This means:
- String models are initialized once at middleware creation
- No runtime model initialization overhead
- All models ready to use when fallback needed

### Exception Handling Strategy

The middleware uses a simple try-except pattern:
1. Catch any exception from model call
2. Store as `last_exception`
3. Try next model
4. If all models fail, re-raise `last_exception`

This approach:
- Preserves original exception context
- Works with any exception type
- Doesn't log or swallow errors
- Simple and predictable behavior

### Request Override

The `request.override(model=fallback_model)` method creates a new request with:
- Same messages
- Same system prompt
- Same tools
- Same other parameters
- Different model

This ensures fallback models receive identical context to primary model.

### No State Tracking

The middleware is stateless:
- No counters or metrics
- No tracking of which models succeeded
- No learning from failures
- Each invocation independent

For tracking, combine with monitoring middleware or external logging.

### Sync and Async Implementations

Both sync and async versions are fully implemented:
- `wrap_model_call`: Synchronous handler
- `awrap_model_call`: Asynchronous handler

The logic is duplicated rather than one delegating to the other, ensuring:
- No async-to-sync conversions
- Proper async exception handling
- Native async/await support

### Eager vs Lazy Initialization

Models are initialized eagerly (during middleware construction):
- Pros: Fast fallback when needed, immediate validation
- Cons: All models loaded even if not used

Alternative design would initialize lazily on first use.

## Related Pages

### Core Middleware Infrastructure
- **langchain-ai_langchain_AgentMiddleware_class.md**: Base middleware interface
- **langchain-ai_langchain_middleware_hooks.txt**: Middleware hook system

### Other V1 Middleware Implementations
- **langchain-ai_langchain_ContextEditingMiddleware.md**: Context window management
- **langchain-ai_langchain_ModelCallLimitMiddleware.md**: Model call quota enforcement
- **langchain-ai_langchain_TodoListMiddleware.md**: Task tracking for agents
- **langchain-ai_langchain_LLMToolEmulator.md**: LLM-based tool emulation

### Related Middleware
- **langchain-ai_langchain_ModelRetryMiddleware.md**: Retry same model with backoff
- **langchain-ai_langchain_ToolRetryMiddleware.md**: Fallback for tool calls

### Agent Creation
- **langchain-ai_langchain_create_agent.md**: Agent creation with middleware support

### Model Management
- **langchain-ai_langchain_init_chat_model.md**: Chat model initialization
- **langchain-ai_langchain_ConfigurableModel.mediawiki**: Model configuration patterns

## Architecture Notes

### Design Philosophy
The middleware follows these principles:
- **Fail forward**: Try alternatives rather than immediate failure
- **Transparent**: Appears as single model to agent
- **Exception-driven**: Uses exceptions for control flow
- **Stateless**: No tracking or learning between invocations

### Comparison with Retry Middleware

**ModelFallbackMiddleware**:
- Different models on failure
- Sequential model substitution
- Immediate retry with different model
- No backoff or delays

**ModelRetryMiddleware**:
- Same model on failure
- Exponential backoff
- Transient error recovery
- Configurable retry attempts

These middlewares are complementary and can be combined.

### Performance Considerations
- All fallback models initialized upfront (memory cost)
- No delays between fallback attempts (fast failure recovery)
- Exception handling has minimal overhead
- Each fallback is full model invocation (not cached)

### Production Deployment Patterns

**High Availability**:
```python
fallback = ModelFallbackMiddleware(
    "openai:gpt-4o",           # Same model, different instance
    "anthropic:claude-sonnet", # Different provider
    "google:gemini-flash",     # Third provider
)
```

**Cost Optimization**:
```python
fallback = ModelFallbackMiddleware(
    "openai:gpt-4o-mini",  # Cheaper, same provider
    "anthropic:haiku",     # Cheapest alternative
)
```

**Capability Preservation**:
```python
fallback = ModelFallbackMiddleware(
    "anthropic:claude-sonnet-4-5-20250929",  # Similar capability
    "openai:gpt-4o",                         # Similar capability
)
# Avoid falling back to much weaker models
```

### Error Types Handled

The middleware handles all exceptions:
- Rate limit errors
- Network errors
- Authentication failures
- Model not found
- Invalid requests
- Service outages

No discrimination between error types - all trigger fallback.

### Limitations

1. **No Error Inspection**: Can't skip fallbacks for certain error types
2. **No Partial Results**: All-or-nothing model calls
3. **No Metrics**: Doesn't track which models work/fail
4. **No Circuit Breaking**: Doesn't avoid known-bad models
5. **No Cost Tracking**: No awareness of cost differences

For advanced patterns, consider extending the middleware or using external monitoring.

## Extension Points

### Custom Fallback Logic

```python
class SmartFallbackMiddleware(ModelFallbackMiddleware):
    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except RateLimitError:
            # Use cheaper model for rate limits
            return handler(request.override(model=self.cheap_model))
        except Exception:
            # Use standard fallback chain
            return super().wrap_model_call(request, handler)
```

### Metrics Integration

```python
class InstrumentedFallbackMiddleware(ModelFallbackMiddleware):
    def wrap_model_call(self, request, handler):
        try:
            result = super().wrap_model_call(request, handler)
            self.metrics.record_success(request.model)
            return result
        except Exception as e:
            self.metrics.record_failure(request.model, type(e))
            raise
```

### Conditional Fallback

```python
class SelectiveFallbackMiddleware(ModelFallbackMiddleware):
    def __init__(self, *models, error_types=(RateLimitError, ServiceUnavailable)):
        super().__init__(*models)
        self.error_types = error_types

    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except self.error_types as e:
            # Only fallback for specific errors
            return super().wrap_model_call(request, handler)
        except Exception:
            raise  # Don't fallback for other errors
```

### Circuit Breaker Pattern

```python
class CircuitBreakerFallback(ModelFallbackMiddleware):
    def __init__(self, *models):
        super().__init__(*models)
        self.failure_counts = defaultdict(int)
        self.circuit_open = set()

    def wrap_model_call(self, request, handler):
        # Skip models with open circuits
        available_models = [
            m for m in self.models
            if m.model_name not in self.circuit_open
        ]
        # Implement circuit breaker logic...
```
