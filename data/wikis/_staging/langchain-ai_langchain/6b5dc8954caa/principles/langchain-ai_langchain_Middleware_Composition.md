{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Middleware Pattern|https://en.wikipedia.org/wiki/Middleware]]
* [[source::Doc|Chain of Responsibility|https://refactoring.guru/design-patterns/chain-of-responsibility]]
* [[source::Doc|LangChain Agents|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Agents]], [[domain::Design_Patterns]], [[domain::Software_Architecture]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Architectural pattern for composing modular interceptors that modify agent behavior at defined lifecycle points.

=== Description ===

Middleware Composition is a design pattern that allows adding cross-cutting concerns to agent execution through composable, reusable components. Instead of hardcoding behaviors like logging, retry logic, or rate limiting into the agent core, these concerns are factored out into middleware that can be mixed and matched.

This pattern addresses several architectural needs:
* **Separation of Concerns:** Business logic stays in tools; operational concerns go in middleware
* **Reusability:** Common patterns (retry, logging, caching) are implemented once
* **Configurability:** Behavior can be customized by changing middleware composition
* **Testability:** Individual middleware can be tested in isolation

Middleware forms a pipeline where each component can:
1. Run code before/after agent lifecycle events
2. Intercept and modify requests/responses
3. Short-circuit execution (skip model calls, return cached results)
4. Extend agent state with additional fields

=== Usage ===

Use Middleware Composition when:
* Adding operational concerns (logging, monitoring, tracing)
* Implementing reliability patterns (retry, fallback, circuit breaker)
* Adding security (PII detection, rate limiting, approval workflows)
* Managing context (summarization, memory, context windows)
* Implementing custom control flow (human-in-the-loop)

Design considerations:
* Middleware order matters (first in list wraps outermost)
* State modifications are merged, not replaced
* Async middleware should be used for async agents

== Theoretical Basis ==

Middleware Composition implements the **Chain of Responsibility** and **Decorator** patterns in the context of agent execution.

'''1. Lifecycle Hook Model'''

Agent execution has defined interception points:

<syntaxhighlight lang="python">
# Pseudo-code for agent lifecycle
def agent_execution(state):
    # Phase 1: Before agent
    for middleware in middlewares:
        state = merge(state, middleware.before_agent(state))

    while not done:
        # Phase 2: Before model
        for middleware in middlewares:
            state = merge(state, middleware.before_model(state))

        # Phase 3: Model call (wrapped)
        response = call_model_with_wrapping(state)

        # Phase 4: After model
        for middleware in middlewares:
            state = merge(state, middleware.after_model(state))

        # Phase 5: Tool calls (wrapped)
        if response.has_tool_calls:
            tool_results = call_tools_with_wrapping(response.tool_calls)
</syntaxhighlight>

'''2. Handler Composition (Onion Model)'''

Wrap methods compose as nested layers:

<syntaxhighlight lang="python">
# Pseudo-code for handler composition
def compose_handlers(middlewares, core_handler):
    handler = core_handler
    for middleware in reversed(middlewares):  # First middleware = outermost
        handler = lambda req, h=handler, m=middleware: m.wrap_model_call(req, h)
    return handler

# Execution flow: M1 -> M2 -> M3 -> Core -> M3 -> M2 -> M1
#                outer          inner          outer
</syntaxhighlight>

'''3. Immutable Request Pattern'''

Requests use immutable override pattern for safety:

<syntaxhighlight lang="python">
@dataclass
class ModelRequest:
    model: BaseChatModel
    messages: list[Message]
    # ... other fields

    def override(self, **kwargs) -> ModelRequest:
        """Return new request with specified fields replaced."""
        return replace(self, **kwargs)

# Usage in middleware:
def wrap_model_call(request, handler):
    # Modify request immutably
    modified = request.override(temperature=0.5)
    return handler(modified)
</syntaxhighlight>

'''4. State Extension Pattern'''

Middleware can extend state schema:

<syntaxhighlight lang="python">
class CustomState(AgentState):
    """Extended state for middleware."""
    retry_count: int = 0
    call_history: list[str] = field(default_factory=list)


class TrackingMiddleware(AgentMiddleware[CustomState, Any]):
    state_schema = CustomState

    def before_model(self, state, runtime):
        return {"call_history": state.call_history + ["model_call"]}
</syntaxhighlight>

'''5. Jump Control Flow'''

Middleware can redirect execution:

<syntaxhighlight lang="python">
def before_model(state, runtime):
    if should_skip_model(state):
        return {"jump_to": "end"}  # Skip to end of agent
    if should_call_tools_directly(state):
        return {"jump_to": "tools"}  # Jump to tool execution
    return None  # Continue normally
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_AgentMiddleware_class]]

=== Used By Workflows ===
* Agent_Creation_Workflow (Step 3)
