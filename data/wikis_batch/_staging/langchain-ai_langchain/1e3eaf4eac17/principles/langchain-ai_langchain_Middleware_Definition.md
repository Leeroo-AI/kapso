= Middleware_Definition =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/middleware/types.py:L330-690
|domains=Agent Architecture, Middleware Systems, Hook Mechanisms
|last_updated=2025-12-17
}}

== Overview ==

'''Middleware_Definition''' establishes the foundational abstraction for intercepting and customizing agent behavior through a class-based or decorator-based pattern. Middleware instances can hook into multiple lifecycle stages of agent execution, enabling modular cross-cutting concerns like authentication, logging, caching, retry logic, and dynamic prompt generation.

This principle defines the contract that all middleware must follow, specifying which methods can be overridden and how middleware composes with the agent execution flow.

== Description ==

The Middleware_Definition principle centers on the '''AgentMiddleware''' base class and its companion decorators that allow developers to inject custom logic at specific points in the agent lifecycle without modifying the core agent implementation.

'''Core Concepts:'''

* '''Lifecycle Hooks''': Middleware can implement methods that execute before/after the agent runs, before/after the model is called, and can wrap model/tool calls
* '''Dual Implementation''': Both synchronous and asynchronous execution paths are supported through paired methods (e.g., `wrap_model_call` / `awrap_model_call`)
* '''State Schema Extension''': Middleware can declare custom state fields through `state_schema` attribute
* '''Tool Registration''': Middleware can contribute additional tools through the `tools` attribute
* '''Decorator Pattern''': Standalone functions can be converted to middleware instances using decorators like `@before_model`, `@wrap_model_call`, etc.

'''Key Properties:'''

* '''Composability''': Multiple middleware instances compose sequentially, with the first middleware in the list acting as the outermost layer
* '''Immutability''': Middleware should not directly mutate state; instead, they return state updates as dictionaries
* '''Opt-in Hooks''': Middleware only needs to implement the hooks relevant to its functionality
* '''Error Handling''': NotImplementedError is raised when middleware lacks the required sync/async implementation for the execution path

== Usage ==

The Middleware_Definition principle is applied when:

* Implementing reusable agent behaviors that apply across different agent configurations
* Adding observability, monitoring, or instrumentation to agent execution
* Implementing conditional flow control (early exits, retries, fallbacks)
* Extending agent state with middleware-specific fields
* Dynamically modifying model requests or responses

'''Pattern Selection:'''

* Use '''class-based middleware''' (subclass AgentMiddleware) when:
** Implementing multiple hooks in a single component
** Maintaining internal state across hook invocations
** Requiring both sync and async implementations
** Registering custom tools or state schemas

* Use '''decorator-based middleware''' when:
** Implementing a single hook function
** Creating simple, focused middleware
** Preferring functional programming style
** Avoiding class boilerplate

== Theoretical Basis ==

The Middleware_Definition principle draws from several software architecture patterns:

'''1. Interceptor Pattern'''

Middleware acts as an interceptor that captures and potentially modifies the flow of execution. This pattern is common in web frameworks (Express.js, Django middleware) and provides a clean separation of concerns.

'''2. Chain of Responsibility'''

Multiple middleware instances form a chain where each middleware can:
* Pass execution to the next handler
* Short-circuit the chain by not calling the handler
* Modify the request before passing it forward
* Transform the response after receiving it from inner layers

'''3. Aspect-Oriented Programming (AOP)'''

Middleware enables cross-cutting concerns (logging, security, caching) to be modularized and applied declaratively rather than woven throughout business logic.

'''4. Decorator Pattern (Structural)'''

The middleware wrapping mechanism implements the structural decorator pattern, where each middleware layer adds behavior while preserving the core interface.

'''5. Strategy Pattern'''

Different middleware implementations provide alternative strategies for handling the same lifecycle events, allowing runtime composition of behaviors.

'''Design Principles:'''

* '''Open/Closed Principle''': Agent core is closed for modification but open for extension through middleware
* '''Single Responsibility''': Each middleware focuses on one cross-cutting concern
* '''Dependency Inversion''': Agent depends on the AgentMiddleware abstraction, not concrete implementations

== Related Pages ==

'''Implementation:'''
* [[langchain-ai_langchain_AgentMiddleware_base]] - Base class implementation and decorator factories

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Hook execution semantics and ordering
* [[langchain-ai_langchain_Middleware_State_Schema]] - State extension mechanism
* [[langchain-ai_langchain_Middleware_Composition_Order]] - Multi-middleware orchestration

'''Implementation Details:'''
* [[langchain-ai_langchain_middleware_hooks]] - Individual hook implementations
* [[langchain-ai_langchain_chain_handlers]] - Handler composition logic

'''Workflows:'''
* [[langchain-ai_langchain_Middleware_Composition]] - Complete middleware workflow

[[Category:Principles]]
[[Category:Agent Architecture]]
[[Category:Middleware Systems]]
