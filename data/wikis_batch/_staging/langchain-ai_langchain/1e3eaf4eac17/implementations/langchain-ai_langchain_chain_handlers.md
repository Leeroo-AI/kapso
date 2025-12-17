= chain_handlers =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/factory.py:L86-196, L431-538
|domains=Handler Composition, Middleware Runtime, Function Chaining
|last_updated=2025-12-17
}}

== Overview ==

'''chain_handlers''' implements the runtime composition logic that combines multiple middleware wrapper hooks (wrap_model_call, wrap_tool_call) into single composed handlers. This implementation creates the onion-style nesting where the first middleware in the list becomes the outermost layer with complete control over inner layers.

== Description ==

The implementation provides four key functions:

'''1. _chain_model_call_handlers (Sync)'''

Composes synchronous `wrap_model_call` handlers:
* Takes list of sync handler functions
* Returns single composed handler (or None if list empty)
* Implements right-to-left composition (last handler is innermost)
* Normalizes return values to ModelResponse

'''2. _chain_async_model_call_handlers (Async)'''

Composes asynchronous `awrap_model_call` handlers:
* Takes list of async handler functions
* Returns single composed async handler (or None if list empty)
* Same composition semantics as sync version
* Handles async/await properly

'''3. _chain_tool_call_wrappers (Sync)'''

Composes synchronous `wrap_tool_call` wrappers:
* Takes list of sync wrapper functions
* Returns single composed wrapper (or None if list empty)
* Implements right-to-left composition

'''4. _chain_async_tool_call_wrappers (Async)'''

Composes asynchronous `awrap_tool_call` wrappers:
* Takes list of async wrapper functions
* Returns single composed async wrapper (or None if list empty)
* Same composition semantics as sync version

'''Composition Strategy:'''

All functions use the same composition algorithm:
1. Handle empty list → return None
2. Handle single handler → return normalized version
3. For multiple handlers:
   - Define `compose_two(outer, inner)` function
   - Compose right-to-left using fold/reduce
   - Return final normalized composed handler

== Code Reference ==

'''Sync Model Call Handler Composition:'''

<source lang="python">
# _chain_model_call_handlers (factory.py lines 86-196)
def _chain_model_call_handlers(
    handlers: Sequence[
        Callable[
            [ModelRequest, Callable[[ModelRequest], ModelResponse]],
            ModelResponse | AIMessage,
        ]
    ],
) -> (
    Callable[
        [ModelRequest, Callable[[ModelRequest], ModelResponse]],
        ModelResponse,
    ]
    | None
):
    """Compose multiple wrap_model_call handlers into single middleware stack.

    Composes handlers so first in list becomes outermost layer. Each handler
    receives a handler callback to execute inner layers.
    """
    if not handlers:
        return None

    if len(handlers) == 1:
        # Single handler - wrap to normalize output
        single_handler = handlers[0]

        def normalized_single(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            result = single_handler(request, handler)
            return _normalize_to_model_response(result)

        return normalized_single

    def compose_two(
        outer: Callable[...],
        inner: Callable[...],
    ) -> Callable[...]:
        """Compose two handlers where outer wraps inner."""

        def composed(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # Create a wrapper that calls inner with the base handler and normalizes
            def inner_handler(req: ModelRequest) -> ModelResponse:
                inner_result = inner(req, handler)
                return _normalize_to_model_response(inner_result)

            # Call outer with the wrapped inner as its handler and normalize
            outer_result = outer(request, inner_handler)
            return _normalize_to_model_response(outer_result)

        return composed

    # Compose right-to-left: outer(inner(innermost(handler)))
    result = handlers[-1]
    for handler in reversed(handlers[:-1]):
        result = compose_two(handler, result)

    # Wrap to ensure final return type is exactly ModelResponse
    def final_normalized(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        final_result = result(request, handler)
        return _normalize_to_model_response(final_result)

    return final_normalized
</source>

'''Async Model Call Handler Composition:'''

<source lang="python">
# _chain_async_model_call_handlers (factory.py lines 198-280)
def _chain_async_model_call_handlers(
    handlers: Sequence[
        Callable[
            [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
            Awaitable[ModelResponse | AIMessage],
        ]
    ],
) -> (
    Callable[
        [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
        Awaitable[ModelResponse],
    ]
    | None
):
    """Compose multiple async `wrap_model_call` handlers into single middleware stack."""
    if not handlers:
        return None

    if len(handlers) == 1:
        single_handler = handlers[0]

        async def normalized_single(
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            result = await single_handler(request, handler)
            return _normalize_to_model_response(result)

        return normalized_single

    def compose_two(outer, inner):
        """Compose two async handlers where outer wraps inner."""

        async def composed(
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            async def inner_handler(req: ModelRequest) -> ModelResponse:
                inner_result = await inner(req, handler)
                return _normalize_to_model_response(inner_result)

            outer_result = await outer(request, inner_handler)
            return _normalize_to_model_response(outer_result)

        return composed

    # Compose right-to-left
    result = handlers[-1]
    for handler in reversed(handlers[:-1]):
        result = compose_two(handler, result)

    async def final_normalized(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        final_result = await result(request, handler)
        return _normalize_to_model_response(final_result)

    return final_normalized
</source>

'''Tool Call Wrapper Composition:'''

<source lang="python">
# _chain_tool_call_wrappers (factory.py lines 431-474)
def _chain_tool_call_wrappers(
    wrappers: Sequence[ToolCallWrapper],
) -> ToolCallWrapper | None:
    """Compose wrappers into middleware stack (first = outermost).

    Args:
        wrappers: Wrappers in middleware order.

    Returns:
        Composed wrapper, or `None` if empty.
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer: ToolCallWrapper, inner: ToolCallWrapper) -> ToolCallWrapper:
        """Compose two wrappers where outer wraps inner."""

        def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], ToolMessage | Command],
        ) -> ToolMessage | Command:
            # Create a callable that invokes inner with the original execute
            def call_inner(req: ToolCallRequest) -> ToolMessage | Command:
                return inner(req, execute)

            # Outer can call call_inner multiple times
            return outer(request, call_inner)

        return composed

    # Chain all wrappers: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result
</source>

'''Async Tool Call Wrapper Composition:'''

<source lang="python">
# _chain_async_tool_call_wrappers (factory.py lines 477-538)
def _chain_async_tool_call_wrappers(
    wrappers: Sequence[
        Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
            Awaitable[ToolMessage | Command],
        ]
    ],
) -> (
    Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
        Awaitable[ToolMessage | Command],
    ]
    | None
):
    """Compose async wrappers into middleware stack (first = outermost)."""
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer, inner):
        """Compose two async wrappers where outer wraps inner."""

        async def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
        ) -> ToolMessage | Command:
            async def call_inner(req: ToolCallRequest) -> ToolMessage | Command:
                return await inner(req, execute)

            return await outer(request, call_inner)

        return composed

    # Chain all wrappers: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result
</source>

'''Normalization Helper:'''

<source lang="python">
# _normalize_to_model_response (factory.py lines 79-83)
def _normalize_to_model_response(result: ModelResponse | AIMessage) -> ModelResponse:
    """Normalize middleware return value to ModelResponse."""
    if isinstance(result, AIMessage):
        return ModelResponse(result=[result], structured_response=None)
    return result
</source>

== I/O Contract ==

'''_chain_model_call_handlers:'''

'''Input:'''
* `handlers: Sequence[Callable]` - List of sync wrap_model_call handlers from middleware
* Each handler signature: `(ModelRequest, handler) -> ModelResponse | AIMessage`

'''Output:'''
* `Callable | None` - Single composed handler with same signature, or None if list empty
* Composed handler normalizes all returns to ModelResponse

'''Processing:'''
1. Empty list → return None
2. Single handler → wrap with normalization, return
3. Multiple handlers:
   - Start with last handler (innermost)
   - For each previous handler (right-to-left):
     * Compose with current result using `compose_two`
   - Wrap final result with normalization
   - Return composed handler

'''_chain_tool_call_wrappers:'''

'''Input:'''
* `wrappers: Sequence[ToolCallWrapper]` - List of sync wrap_tool_call wrappers
* Each wrapper signature: `(ToolCallRequest, handler) -> ToolMessage | Command`

'''Output:'''
* `ToolCallWrapper | None` - Single composed wrapper, or None if list empty

'''Processing:'''
1. Empty list → return None
2. Single wrapper → return as-is
3. Multiple wrappers:
   - Start with last wrapper (innermost)
   - For each previous wrapper (right-to-left):
     * Compose with current result
   - Return composed wrapper

'''Composition Semantics:'''

Given handlers `[A, B, C]`, composition creates:

<pre>
def composed(request, base_handler):
    # A is outermost
    def handler_for_A(req):
        # B is middle layer
        def handler_for_B(req):
            # C is innermost
            def handler_for_C(req):
                return base_handler(req)  # Actual execution
            return C(req, handler_for_C)
        return B(req, handler_for_B)
    return A(request, handler_for_A)
</pre>

Execution flow:
1. A called with request and handler_for_A
2. A calls handler_for_A, which invokes B
3. B called with request and handler_for_B
4. B calls handler_for_B, which invokes C
5. C called with request and handler_for_C
6. C calls handler_for_C, which calls base_handler (actual model/tool)
7. Result propagates back: base → C → B → A → caller

== Usage Examples ==

'''Example 1: Composed Model Call Handlers'''

<source lang="python">
from langchain.agents.middleware.types import wrap_model_call

@wrap_model_call
def auth_handler(request, handler):
    """Outermost: Check authentication."""
    if not is_authenticated(request.runtime.context):
        raise PermissionError("Not authenticated")
    return handler(request)

@wrap_model_call
def logging_handler(request, handler):
    """Middle: Log request/response."""
    print(f"Calling model: {request.model}")
    response = handler(request)
    print(f"Model returned: {len(response.result)} messages")
    return response

@wrap_model_call
def retry_handler(request, handler):
    """Innermost: Retry on failure."""
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}: {e}")

# Order matters: auth → logging → retry → model
agent = create_agent(
    model="openai:gpt-4",
    middleware=[auth_handler, logging_handler, retry_handler]
)

# Execution flow:
# 1. auth_handler checks authentication
# 2. auth_handler calls handler → logging_handler
# 3. logging_handler logs and calls handler → retry_handler
# 4. retry_handler tries calling handler → actual model
# 5. If model fails, retry_handler retries (up to 3 times)
# 6. Result flows back: model → retry → logging → auth → caller
</source>

'''Example 2: Short-Circuit in Composition'''

<source lang="python">
from langchain.agents.middleware.types import wrap_model_call, ModelResponse, AIMessage

@wrap_model_call
def cache_handler(request, handler):
    """Outermost: Check cache first."""
    cache_key = hash_request(request)
    if cached := get_cache(cache_key):
        print("Cache hit - skipping inner handlers")
        return ModelResponse(result=[cached], structured_response=None)

    # Cache miss - call inner handlers
    response = handler(request)
    save_cache(cache_key, response.result[0])
    return response

@wrap_model_call
def expensive_handler(request, handler):
    """This runs only on cache miss."""
    print("Doing expensive preprocessing")
    modified_request = expensive_preprocess(request)
    return handler(modified_request)

# cache_handler can short-circuit expensive_handler
agent = create_agent(
    model="openai:gpt-4",
    middleware=[cache_handler, expensive_handler]
)

# On cache hit: cache_handler returns directly, expensive_handler never runs
# On cache miss: cache_handler → expensive_handler → model
</source>

'''Example 3: Multiple Calls in Composition'''

<source lang="python">
from langchain.agents.middleware.types import wrap_model_call

@wrap_model_call
def voting_handler(request, handler):
    """Outermost: Call handler multiple times and vote."""
    responses = []

    # Call inner handlers 3 times
    for i in range(3):
        print(f"Vote round {i + 1}")
        response = handler(request)
        responses.append(response)

    # Majority vote on responses
    voted_response = majority_vote(responses)
    return voted_response

@wrap_model_call
def randomize_handler(request, handler):
    """Add randomness to each call."""
    modified_request = add_randomness(request)
    return handler(modified_request)

# voting_handler calls inner stack 3 times
agent = create_agent(
    model="openai:gpt-4",
    middleware=[voting_handler, randomize_handler]
)

# Execution:
# voting_handler calls handler (which is randomize_handler) 3 times
# Each call: voting → randomize → model
# voting_handler collects 3 responses and returns voted result
</source>

'''Example 4: Error Boundary in Composition'''

<source lang="python">
from langchain.agents.middleware.types import wrap_model_call, ModelResponse, AIMessage

@wrap_model_call
def error_boundary(request, handler):
    """Outermost: Catch all errors from inner layers."""
    try:
        return handler(request)
    except Exception as e:
        print(f"Error caught: {e}")
        # Return graceful fallback
        return ModelResponse(
            result=[AIMessage(content="Service temporarily unavailable")],
            structured_response=None
        )

@wrap_model_call
def flaky_middleware(request, handler):
    """Might throw exceptions."""
    if random.random() < 0.5:
        raise RuntimeError("Random failure")
    return handler(request)

# error_boundary catches exceptions from all inner layers
agent = create_agent(
    model="openai:gpt-4",
    middleware=[error_boundary, flaky_middleware]
)

# Even if flaky_middleware or model fails, error_boundary returns graceful response
</source>

'''Example 5: Composition with State Tracking'''

<source lang="python">
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse

class TrackerMiddleware(AgentMiddleware):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.call_count = 0

    def wrap_model_call(self, request, handler):
        """Track calls at this layer."""
        self.call_count += 1
        print(f"{self.name} handler: call #{self.call_count}")

        response = handler(request)

        print(f"{self.name} handler: returning")
        return response

# Create 3 trackers
tracker_a = TrackerMiddleware("A")
tracker_b = TrackerMiddleware("B")
tracker_c = TrackerMiddleware("C")

agent = create_agent(
    model="openai:gpt-4",
    middleware=[tracker_a, tracker_b, tracker_c]
)

# Single invocation output:
# A handler: call #1
# B handler: call #1
# C handler: call #1
# C handler: returning
# B handler: returning
# A handler: returning

# Shows onion composition: A → B → C → model → C → B → A
</source>

'''Example 6: Async Composition'''

<source lang="python">
from langchain.agents.middleware.types import wrap_model_call
import asyncio

@wrap_model_call
async def async_timeout(request, handler):
    """Outermost: Enforce timeout on entire chain."""
    try:
        response = await asyncio.wait_for(handler(request), timeout=30.0)
        return response
    except asyncio.TimeoutError:
        print("Request timed out")
        return ModelResponse(result=[AIMessage(content="Request timed out")])

@wrap_model_call
async def async_cache(request, handler):
    """Middle: Async cache lookup."""
    cache_key = hash_request(request)
    if cached := await async_cache_get(cache_key):
        return cached

    response = await handler(request)
    await async_cache_set(cache_key, response)
    return response

@wrap_model_call
async def async_rate_limit(request, handler):
    """Innermost: Rate limiting."""
    await rate_limiter.acquire()
    return await handler(request)

# Async composition: timeout → cache → rate_limit → model
agent = create_agent(
    model="openai:gpt-4",
    middleware=[async_timeout, async_cache, async_rate_limit]
)

# Use with async invocation
response = await agent.ainvoke({"messages": [...]})
</source>

== Related Pages ==

'''Principle:'''
* [[langchain-ai_langchain_Middleware_Composition_Order]] - Composition ordering semantics

'''Related Implementations:'''
* [[langchain-ai_langchain_middleware_hooks]] - Individual hook implementations
* [[langchain-ai_langchain_AgentMiddleware_base]] - Middleware base class

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Middleware abstraction
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Hook types and execution

'''Workflows:'''
* [[langchain-ai_langchain_Middleware_Composition]] - Complete workflow

[[Category:Implementations]]
[[Category:Handler Composition]]
[[Category:Middleware Runtime]]
