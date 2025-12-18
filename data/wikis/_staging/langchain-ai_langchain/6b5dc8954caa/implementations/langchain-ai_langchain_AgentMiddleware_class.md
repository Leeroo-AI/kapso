{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Agents|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Agents]], [[domain::Middleware]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for composing agent lifecycle hooks and interceptors, provided by LangChain's agent middleware system.

=== Description ===

`AgentMiddleware` is the base class for creating middleware that intercepts and modifies agent behavior at various lifecycle points. It provides a comprehensive hook system for customizing agent execution without modifying core agent logic.

Available hooks:
* **before_agent / abefore_agent:** Run before agent execution starts
* **before_model / abefore_model:** Run before each model call
* **after_model / aafter_model:** Run after each model call
* **wrap_model_call / awrap_model_call:** Intercept model execution with handler pattern
* **wrap_tool_call / awrap_tool_call:** Intercept tool execution with handler pattern

Middleware can also register additional tools and define custom state schemas.

=== Usage ===

Use `AgentMiddleware` when:
* Implementing retry logic for model calls
* Adding logging/monitoring to agent execution
* Implementing rate limiting or call limits
* Adding context management (summarization, editing)
* Implementing human-in-the-loop approval
* Adding PII detection/redaction

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/agents/middleware/types.py
* '''Lines:''' L330-688

=== Signature ===
<syntaxhighlight lang="python">
class AgentMiddleware(Generic[StateT, ContextT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the defined methods to customize agent behavior
    between steps in the main agent loop.
    """

    state_schema: type[StateT] = AgentState
    """The schema for state passed to the middleware nodes."""

    tools: list[BaseTool]
    """Additional tools registered by the middleware."""

    @property
    def name(self) -> str:
        """The name of the middleware instance."""

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the agent execution starts."""

    async def abefore_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Async logic to run before the agent execution starts."""

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the model is called."""

    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Async logic to run before the model is called."""

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called."""

    async def aafter_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Async logic to run after the model is called."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept and control model execution via handler callback."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Async intercept and control model execution via handler callback."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallWrapper],
    ) -> ToolCallWrapper:
        """Intercept and control tool execution via handler callback."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolCallWrapper]],
    ) -> ToolCallWrapper:
        """Async intercept and control tool execution via handler callback."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents.middleware import AgentMiddleware
# Or with types:
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Hook Methods) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| state || StateT (AgentState) || Yes || Current agent state with messages and optional fields
|-
| runtime || Runtime[ContextT] || Yes || Runtime context with config and store access
|}

=== Outputs (Hook Methods) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || dict[str, Any] | None || State updates to merge, or None for no changes
|}

=== Inputs (Wrap Methods) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| request || ModelRequest | ToolCallRequest || Yes || Request object with all call details
|-
| handler || Callable || Yes || Handler to execute (call to continue, skip to short-circuit)
|}

=== Outputs (Wrap Methods) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || ModelCallResult | ToolCallWrapper || Response from handler or custom response
|}

== Usage Examples ==

=== Subclass-Based Middleware ===
<syntaxhighlight lang="python">
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
import time


class LoggingMiddleware(AgentMiddleware):
    """Middleware that logs all model calls."""

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        print(f"Calling model with {len(request.messages)} messages")
        start = time.time()

        response = handler(request)

        elapsed = time.time() - start
        print(f"Model responded in {elapsed:.2f}s")
        return response
</syntaxhighlight>

=== Retry Middleware ===
<syntaxhighlight lang="python">
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse


class RetryMiddleware(AgentMiddleware):
    """Middleware that retries failed model calls."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return handler(request)
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {e}")
        raise last_error
</syntaxhighlight>

=== Decorator-Based Middleware ===
<syntaxhighlight lang="python">
from langchain.agents.middleware import before_model, wrap_model_call


@before_model
def add_context(state, runtime):
    """Add context before each model call."""
    return {"context": "Additional context injected"}


@wrap_model_call
def rate_limit(request, handler):
    """Simple rate limiting."""
    import time
    time.sleep(0.1)  # 100ms delay between calls
    return handler(request)
</syntaxhighlight>

=== Using Middleware in Agent ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=[...],
    middleware=[
        LoggingMiddleware(),
        RetryMiddleware(max_retries=3),
    ],
)

result = agent.invoke({"messages": [...]})
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Middleware_Composition]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
