# Middleware Configuration

**Sources:**
- `libs/langchain_v1/langchain/agents/middleware/types.py:L330-690`
- LangChain Documentation: Agent Middleware
- Usage examples in integration tests

**Domains:** Agent Architecture, Execution Hooks, State Management

**Last Updated:** 2025-12-17

---

## Overview

Middleware Configuration is the principle of enabling extensible, composable customization of agent behavior through lifecycle hooks and execution interception. This principle allows developers to inject custom logic at specific points in the agent execution loop without modifying the core agent implementation.

## Description

The principle addresses the challenge of making agent systems extensible and customizable while maintaining a clean separation of concerns. Middleware Configuration establishes a plugin architecture where:

1. **Lifecycle hooks** - Entry points before/after major agent operations (model calls, tool execution, agent start/end)
2. **Execution interception** - Ability to wrap and control model and tool calls (retry logic, caching, modification)
3. **State extensions** - Custom state fields that middleware needs to track across agent execution
4. **Tool registration** - Middleware can contribute additional tools to the agent's capabilities

This principle enables powerful patterns such as:
- Authentication and authorization checks before tool execution
- Caching and memoization of model responses
- Retry logic with exponential backoff
- Logging, tracing, and observability
- Dynamic tool selection based on context
- Rate limiting and quota enforcement
- Context injection (RAG, memory, user profiles)

### Key Architectural Decisions

**Hook-Based vs. Interception-Based**
The system provides both:
- **Simple hooks** (`before_model`, `after_model`) for state observation and modification
- **Interception wrappers** (`wrap_model_call`, `wrap_tool_call`) for full control over execution including short-circuiting, retries, and result transformation

**Middleware Composition Order**
Multiple middleware compose in the order they're provided to `create_agent()`, with the first middleware in the list forming the outermost layer of wrapping. This gives predictable precedence for cross-cutting concerns.

**State Schema Merging**
Each middleware can declare its own state schema (via `state_schema` attribute), and all schemas are merged at agent construction time. This enables middleware to add custom state fields without conflicts.

**Sync/Async Dual Implementation**
Middleware can implement either sync methods (`before_model`, `wrap_model_call`) or async methods (`abefore_model`, `awrap_model_call`), or both. The framework routes to the appropriate implementation based on how the agent is invoked.

**Runtime Access**
Middleware hooks receive both `state` (current agent state) and `runtime` (context like user info, checkpointer, store) enabling both stateful and contextual logic.

## Theoretical Basis

This principle implements several established software patterns:

**Aspect-Oriented Programming (AOP)**
Middleware represents cross-cutting concerns that apply across many execution paths without duplicating code. Like AOP "aspects," middleware intercepts execution at defined join points (lifecycle hooks).

**Chain of Responsibility Pattern**
Multiple middleware form a chain where each can handle, modify, or pass through execution. The `wrap_model_call` composition is a direct implementation of this pattern.

**Decorator Pattern**
Middleware decorates the core agent behavior with additional functionality, wrapping execution without modifying the base implementation.

**Middleware Pattern (Web Frameworks)**
Directly inspired by middleware in web frameworks (Express.js, Django), where request/response processing passes through a series of composable handlers.

**Open/Closed Principle**
Agents are open for extension (via middleware) but closed for modification (core logic remains unchanged).

## Usage

### When to Apply This Principle

Apply Middleware Configuration when:

- Implementing cross-cutting concerns (logging, auth, rate limiting) that apply to many operations
- Adding optional capabilities that users can enable/disable without code changes
- Building reusable agent extensions that can be shared across projects
- Implementing retry logic, circuit breakers, or other resilience patterns
- Injecting dynamic context (RAG, memory) into agent execution
- Creating observability and debugging tooling

### When to Use Alternative Approaches

Consider alternatives when:

- **One-off logic**: Simple custom logic that only applies to a single agent might be better inline
- **Core behavior changes**: Fundamentally changing agent logic might warrant a custom agent implementation
- **Performance critical paths**: Middleware adds function call overhead; for ultra-low-latency needs, consider direct implementation
- **Complex stateful orchestration**: Very complex multi-step workflows might be better as explicit state machines

### Anti-Patterns to Avoid

1. **Overly complex middleware**: Middleware that implements business logic rather than cross-cutting concerns
2. **Hidden state mutations**: Middleware that modifies global state or has side effects not reflected in state updates
3. **Blocking operations**: Long-running synchronous operations in middleware that block agent execution
4. **Order-dependent middleware**: Middleware that only works in a specific order breaks composability
5. **Middleware for tool logic**: Tool-specific logic should be in the tool itself, not middleware
6. **Stateful middleware without schema**: Tracking state without declaring it in `state_schema`

### Best Practices

**Keep Middleware Focused**
Each middleware should address a single concern. Compose multiple small middleware rather than creating monolithic ones.

**Declare State Requirements**
If middleware needs to track state across invocations, explicitly declare state fields via `state_schema`.

**Handle Both Sync and Async**
If your agent might be used in both sync and async contexts, implement both method variants.

**Document Middleware Behavior**
Clearly document what hooks a middleware implements and what effects it has on execution.

**Test Middleware in Isolation**
Write unit tests for middleware logic separate from agent tests.

## Related Pages

**Implementation:**
- [[implemented_by::Implementation:langchain-ai_langchain_AgentMiddleware_class]] - Base class and implementation details

**Related Principles:**
- [[langchain-ai_langchain_Agent_Graph_Construction]] - How middleware is incorporated into agents
- [[langchain-ai_langchain_Tool_Definition]] - Middleware can register tools

**Used In Workflows:**
- [[langchain-ai_langchain_Agent_Creation_and_Execution]] - Middleware configuration is Step 3
- [[langchain-ai_langchain_Middleware_Composition]] - Detailed workflow for middleware

**Related Implementations:**
- [[langchain-ai_langchain_wrap_model_call]] - Model call interception implementation
- [[langchain-ai_langchain_wrap_tool_call]] - Tool call interception implementation
- [[langchain-ai_langchain_before_model]] - Simple lifecycle hook decorator
- [[langchain-ai_langchain_chain_handlers]] - Middleware composition logic

**Environment:**
- [[langchain-ai_langchain_Python]] - Python runtime context
