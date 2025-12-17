# Workflow: Middleware Composition

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Middleware|https://docs.langchain.com/oss/python/langchain/middleware]]
|-
! Domains
| [[domain::LLMs]], [[domain::Agents]], [[domain::Middleware]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Process for composing and applying middleware to customize agent behavior at various lifecycle points.

=== Description ===
This workflow describes how to create, configure, and compose middleware for LangChain agents. The middleware architecture provides hooks at key points in the agent loop: before/after the entire agent execution, before/after each model call, and around tool execution. Middleware can intercept requests/responses, implement retry logic, add approval workflows, manage context windows, and extend agent capabilities.

=== Usage ===
Execute this workflow when you need to:
* Add retry logic with exponential backoff for model/tool calls
* Implement human-in-the-loop approval workflows
* Enforce rate limits or tool call budgets
* Manage context window size via summarization or truncation
* Add logging, monitoring, or observability
* Implement model fallback strategies

== Execution Steps ==

=== Step 1: Define Middleware Class or Decorator ===
[[step::Principle:langchain-ai_langchain_Middleware_Definition]]

Create middleware by subclassing `AgentMiddleware` or using function decorators (`@before_model`, `@after_model`, `@wrap_model_call`, `@wrap_tool_call`). Choose the approach based on complexity: decorators for simple single-hook middleware, classes for complex multi-hook middleware with state.

'''Key considerations:'''
* Decorators create middleware instances automatically
* Class-based middleware can define custom state schema
* Both sync and async versions of hooks are supported

=== Step 2: Implement Lifecycle Hooks ===
[[step::Principle:langchain-ai_langchain_Middleware_Lifecycle_Hooks]]

Implement the appropriate hooks for your use case. Available hooks include:
* `before_agent` / `after_agent`: Run once at start/end of agent execution
* `before_model` / `after_model`: Run before/after each model call in the loop
* `wrap_model_call`: Wrap model execution with custom logic (retry, fallback)
* `wrap_tool_call`: Wrap tool execution with custom logic

'''Key considerations:'''
* `wrap_*` hooks receive a handler callback to execute the inner operation
* Hooks can return state updates or `Command` objects for flow control
* `can_jump_to` decorator config enables conditional graph edges

=== Step 3: Define State Extensions (Optional) ===
[[step::Principle:langchain-ai_langchain_Middleware_State_Schema]]

If your middleware needs to track state across calls, define a custom state schema by setting `state_schema` on your middleware class. State schemas are merged across all middleware when the agent is created.

'''Key considerations:'''
* State schemas must extend `AgentState` TypedDict
* Use `OmitFromInput` / `OmitFromOutput` annotations for internal fields
* State is available via `request.state` or passed to hooks directly

=== Step 4: Register Tools (Optional) ===
[[step::Principle:langchain-ai_langchain_Middleware_Tool_Registration]]

Middleware can provide additional tools to the agent by setting the `tools` attribute. This is useful for middleware that adds capabilities like file search, shell execution, or task tracking.

'''Key considerations:'''
* Tools are merged with user-provided tools
* Middleware tools can be context-aware via runtime access
* Tool names must be unique across all sources

=== Step 5: Configure Composition Order ===
[[step::Principle:langchain-ai_langchain_Middleware_Composition_Order]]

Pass middleware to `create_agent()` in the desired order. For `wrap_*` hooks, the first middleware in the list is the outermost layer (processes first/last). For lifecycle hooks, middleware nodes are added to the graph in order.

'''What happens:'''
* Handlers are composed: `[auth, retry]` means auth wraps retry
* Request flows: auth -> retry -> model
* Response flows: model -> retry -> auth
* Each middleware can short-circuit by not calling the handler

== Execution Diagram ==
{{#mermaid:graph TD
    A[Define Middleware Class/Decorator] --> B[Implement Lifecycle Hooks]
    B --> C[Define State Extensions]
    C --> D[Register Tools]
    D --> E[Configure Composition Order]
}}

== Related Pages ==
* [[step::Principle:langchain-ai_langchain_Middleware_Definition]]
* [[step::Principle:langchain-ai_langchain_Middleware_Lifecycle_Hooks]]
* [[step::Principle:langchain-ai_langchain_Middleware_State_Schema]]
* [[step::Principle:langchain-ai_langchain_Middleware_Tool_Registration]]
* [[step::Principle:langchain-ai_langchain_Middleware_Composition_Order]]
