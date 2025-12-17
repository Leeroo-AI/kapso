= Middleware_Lifecycle_Hooks =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/middleware/types.py:L351-690
|domains=Agent Execution Flow, Hook Systems, Lifecycle Management
|last_updated=2025-12-17
}}

== Overview ==

'''Middleware_Lifecycle_Hooks''' defines the execution semantics and timing of six distinct interception points in the agent execution lifecycle. These hooks enable middleware to observe and modify agent behavior at precise moments: before/after the entire agent run, before/after individual model calls, and around model/tool invocations.

This principle establishes the contract for when hooks execute, what they can observe, what they can modify, and how they compose with multiple middleware instances.

== Description ==

The principle identifies six fundamental lifecycle hooks, organized into three categories:

'''1. Agent-Level Hooks (Once-Per-Execution)'''

* '''before_agent''': Executes once at the start of agent execution, before entering the main loop
** Use cases: Initialize session state, validate inputs, load context
** Runs: Exactly once per agent invocation
** Position: Outermost layer, before any model interaction

* '''after_agent''': Executes once at the end of agent execution, after exiting the main loop
** Use cases: Cleanup resources, log final results, persist state
** Runs: Exactly once per agent invocation
** Position: Outermost layer, after all model/tool interactions complete

'''2. Model-Level Hooks (Once-Per-Loop-Iteration)'''

* '''before_model''': Executes before each model call in the agent loop
** Use cases: Modify message history, inject context, conditional routing
** Runs: Once per agent loop iteration, before model invocation
** Position: Just before model node execution
** Special capability: Can set `jump_to` to short-circuit to tools/end

* '''after_model''': Executes after each model call in the agent loop
** Use cases: Validate model output, log responses, trigger actions
** Runs: Once per agent loop iteration, after model returns
** Position: Immediately after model node execution
** Special capability: Can set `jump_to` to repeat model call or exit early

'''3. Execution Wrapper Hooks (Intercept-And-Control)'''

* '''wrap_model_call''': Wraps the model invocation with full execution control
** Use cases: Retry logic, fallback models, caching, response rewriting
** Runs: Once per model call, with control over handler invocation
** Position: Wraps the model execution itself
** Special capability: Can call handler multiple times, skip it, or modify request/response

* '''wrap_tool_call''': Wraps individual tool invocations with full execution control
** Use cases: Tool retry logic, argument modification, result caching, monitoring
** Runs: Once per tool call, with control over tool execution
** Position: Wraps each individual tool execution
** Special capability: Can call handler multiple times, skip it, or modify tool request/response

'''Hook Execution Order (Single Iteration):'''

<pre>
1. before_agent (first iteration only)
2. before_model
3. wrap_model_call (outer -> inner)
   └─> Model execution
4. after_model
5. [if tools needed]
   wrap_tool_call (outer -> inner, per tool)
   └─> Tool execution
6. [loop back to step 2 or exit]
7. after_agent (final iteration only)
</pre>

'''Key Properties:'''

* '''Composability''': Multiple middleware hooks of the same type chain sequentially
* '''State Access''': All hooks receive the current agent state and runtime context
* '''State Updates''': Hooks return dictionaries that merge into agent state
* '''Flow Control''': Hooks can return `{"jump_to": "end"|"model"|"tools"}` to alter routing
* '''Async Support''': Each hook has sync and async variants (e.g., `before_model` / `abefore_model`)

== Usage ==

The Middleware_Lifecycle_Hooks principle is applied when:

* Timing matters for middleware behavior (e.g., initialization vs. cleanup)
* Middleware needs to observe or modify specific stages of execution
* Implementing conditional flow control based on agent state
* Coordinating behavior across multiple middleware instances

'''Hook Selection Guidance:'''

* Use '''before_agent''' for:
** One-time initialization (loading user preferences, auth checks)
** Global state setup that applies to the entire conversation
** Input validation that prevents agent execution

* Use '''before_model''' for:
** Per-iteration state updates (turn counting, context injection)
** Dynamic message list manipulation
** Conditional early exit before expensive model call

* Use '''wrap_model_call''' for:
** Retry logic with exponential backoff
** Model fallback chains (try GPT-4, fallback to GPT-3.5)
** Response caching and deduplication
** Response content rewriting or filtering

* Use '''after_model''' for:
** Model output validation
** Conditional routing based on response content
** Triggering side effects (logging, metrics, notifications)

* Use '''wrap_tool_call''' for:
** Tool-specific retry logic
** Tool call argument validation/modification
** Tool result caching
** Tool execution monitoring and timeouts

* Use '''after_agent''' for:
** Saving conversation history to database
** Computing and storing session metrics
** Resource cleanup (closing connections, releasing locks)
** Final output transformation

== Theoretical Basis ==

The Middleware_Lifecycle_Hooks principle draws from several execution model concepts:

'''1. Servlet Filter Chain (Java)'''

The before/after hook pattern mirrors servlet filters, where requests pass through a chain of filters before reaching the servlet, and responses pass back through the chain in reverse order.

'''2. Middleware Pipeline (Express.js, ASP.NET)'''

The execution flow follows the middleware pipeline pattern where:
* Request flows forward through middleware (before hooks)
* Core logic executes (model/tool calls)
* Response flows backward through middleware (after hooks)

'''3. Around Advice (Aspect-Oriented Programming)'''

The `wrap_model_call` and `wrap_tool_call` hooks implement "around advice" from AOP, providing complete control over the join point execution.

'''4. Template Method Pattern'''

The agent execution flow acts as a template method with defined hook points where subclasses (middleware) can inject custom behavior.

'''5. Observer Pattern'''

The before/after hooks implement a form of the observer pattern where middleware observes and reacts to lifecycle events without modifying the core agent logic.

'''Design Principles:'''

* '''Separation of Concerns''': Each hook serves a distinct purpose in the lifecycle
* '''Least Privilege''': Hooks only receive the data they need (state, runtime, request)
* '''Composability''': Hooks can be arbitrarily chained without interference
* '''Predictability''': Hook execution order is deterministic and well-defined

'''Execution Model Properties:'''

* '''Before Hooks''': Execute in middleware list order (index 0 first)
* '''After Hooks''': Execute in reverse middleware list order (last middleware first)
* '''Wrapper Hooks''': Compose onion-style (first middleware = outermost layer)
* '''Async Handling''': Sync and async hooks are selected based on invocation method

== Related Pages ==

'''Implementation:'''
* [[langchain-ai_langchain_middleware_hooks]] - Concrete hook implementations

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Base middleware abstraction
* [[langchain-ai_langchain_Middleware_Composition_Order]] - Multi-middleware orchestration
* [[langchain-ai_langchain_Middleware_State_Schema]] - State management across hooks

'''Implementation Details:'''
* [[langchain-ai_langchain_AgentMiddleware_base]] - Hook method definitions
* [[langchain-ai_langchain_chain_handlers]] - Handler composition for wrappers

'''Workflows:'''
* [[langchain-ai_langchain_Middleware_Composition]] - Complete workflow

[[Category:Principles]]
[[Category:Agent Execution]]
[[Category:Lifecycle Management]]
