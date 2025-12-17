= Middleware_Composition_Order =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/factory.py:L86-196, L431-538
|domains=Middleware Composition, Execution Order, Handler Chaining
|last_updated=2025-12-17
}}

== Overview ==

'''Middleware_Composition_Order''' defines the deterministic ordering and composition semantics when multiple middleware instances are present. This principle ensures predictable behavior when middleware layers stack, specifying which middleware executes first, how wrapper hooks compose, and how the execution flow propagates through the middleware chain.

== Description ==

The principle establishes clear rules for middleware composition across three dimensions:

'''1. Middleware List Order'''

The order of middleware in the `create_agent(middleware=[...])` list determines execution priority:

<source lang="python">
agent = create_agent(
    model="openai:gpt-4",
    middleware=[auth, logging, caching]  # Order matters!
)
</source>

* '''List position 0''' (first) = Outermost layer
* '''List position N''' (last) = Innermost layer

'''2. Hook Type Execution Pattern'''

Different hook types have different execution orders:

'''Before Hooks (before_agent, before_model):'''
* Execute in '''forward order''' (index 0 → N)
* First middleware in list executes first
* Pattern: auth → logging → caching → model

'''After Hooks (after_model, after_agent):'''
* Execute in '''reverse order''' (index N → 0)
* Last middleware in list executes first
* Pattern: model → caching → logging → auth

'''Wrapper Hooks (wrap_model_call, wrap_tool_call):'''
* Compose in '''onion/nesting order'''
* First middleware in list becomes outermost layer
* Execution: outer calls inner, which calls innermost, then returns propagate back
* Pattern: auth(logging(caching(model)))

'''3. Composition Semantics'''

'''Sequential Composition (Before/After Hooks):'''

<pre>
middleware = [A, B, C]

before_agent flow:
START → A.before_agent → B.before_agent → C.before_agent → before_model hooks

after_agent flow:
after_model hooks → C.after_agent → B.after_agent → A.after_agent → END
</pre>

* Each hook runs to completion before next hook starts
* State updates from each hook merge sequentially
* Later hooks see state updates from earlier hooks

'''Nested Composition (Wrapper Hooks):'''

<pre>
middleware = [A, B, C]

wrap_model_call flow:
A.wrap_model_call(request, handler_A)
  ├─> handler_A calls B.wrap_model_call(request, handler_B)
      ├─> handler_B calls C.wrap_model_call(request, handler_C)
          ├─> handler_C calls actual model
          └─> returns response to C
      └─> C processes response, returns to B
  └─> B processes response, returns to A
└─> A processes response, returns to caller

Result: A wraps B wraps C wraps model
</pre>

* Outer middleware controls when/if inner middleware executes
* Outer middleware can call handler multiple times (retry)
* Outer middleware can skip handler (short-circuit)
* Responses propagate back through layers

'''4. Execution Order Guarantees'''

* '''Determinism''': Same middleware list produces same execution order
* '''Isolation''': Middleware at position i cannot affect execution order of middleware at position j
* '''Predictability''': Developers can reason about execution flow from middleware list order
* '''Composability''': Any combination of middleware composes correctly

== Usage ==

The Middleware_Composition_Order principle is applied when:

* Designing middleware that depends on other middleware having run first
* Debugging middleware interaction issues
* Optimizing middleware order for performance
* Ensuring security/validation middleware runs before business logic

'''Order Selection Guidelines:'''

'''Authentication/Authorization Middleware:'''
* Position: '''First''' (index 0)
* Reason: Must verify credentials before any other middleware runs
* Example: `[auth, logging, business_logic]`

'''Input Validation Middleware:'''
* Position: '''First''' or '''Early'''
* Reason: Prevent invalid data from propagating through middleware chain
* Example: `[validation, transformation, business_logic]`

'''Logging/Monitoring Middleware:'''
* Position: '''Early''' (after auth/validation)
* Reason: Log after security checks pass but capture all subsequent operations
* Example: `[auth, logging, business_logic]`

'''Caching Middleware:'''
* Position: '''Variable''' based on what should be cached
* Before business logic: Cache final results
* After transformation: Cache raw data
* Example: `[auth, logging, caching, expensive_operation]`

'''Error Handling Middleware:'''
* Position: '''Outermost''' (index 0) for wrapper hooks
* Reason: Catch errors from all inner layers
* Example: `[error_handler, business_logic, data_access]`

'''Response Transformation:'''
* Position: '''Late''' or '''Last'''
* Reason: Transform after all business logic completes
* Example: `[business_logic, formatter, serializer]`

== Theoretical Basis ==

The Middleware_Composition_Order principle draws from several composition models:

'''1. Functional Composition'''

Wrapper hooks implement mathematical function composition:

<pre>
f(g(h(x))) where:
- f = outermost middleware
- g = middle middleware
- h = innermost middleware
- x = model/tool call
</pre>

Properties:
* Associative: (f ∘ g) ∘ h = f ∘ (g ∘ h)
* Order-dependent: f ∘ g ≠ g ∘ f (generally)

'''2. Aspect-Oriented Programming (AOP)'''

Before/after hooks implement AOP advice ordering:
* Multiple before-advice execute in declaration order
* Multiple after-advice execute in reverse declaration order
* Around-advice (wrappers) nest with first declared as outermost

'''3. Chain of Responsibility'''

Wrapper hooks implement CoR pattern:
* Each handler decides whether to pass request to next handler
* Handlers can transform request/response
* Chain order determines handling sequence

'''4. Onion Architecture'''

Wrapper composition follows onion/layered architecture:
* Outer layers wrap inner layers
* Request flows inward through layers
* Response flows outward through layers
* Core (model/tool) is at center

'''5. Decorator Pattern (Behavioral)'''

Middleware composition implements behavioral decorator stacking:
* Each decorator adds behavior
* Decorators are transparent to core component
* Order of decoration affects behavior

'''Composition Laws:'''

* '''Identity''': Empty middleware list = no modification
* '''Left Identity''': [identity_middleware, m] ≈ [m]
* '''Right Identity''': [m, identity_middleware] ≈ [m]
* '''Associativity''': Grouping doesn't affect result (but order does)

'''Order Dependency Analysis:'''

* '''Commutative''': Order doesn't matter
** Example: Independent logging middleware
** Rare in practice

* '''Partially Ordered''': Some ordering constraints required
** Example: Auth before business logic, but logging position flexible
** Most common scenario

* '''Totally Ordered''': Strict ordering required
** Example: Validation → Transformation → Encryption
** Security-critical scenarios

== Related Pages ==

'''Implementation:'''
* [[langchain-ai_langchain_chain_handlers]] - Handler composition implementation

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Base middleware abstraction
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Hook execution semantics
* [[langchain-ai_langchain_Middleware_State_Schema]] - State flow through composition
* [[langchain-ai_langchain_Middleware_Tool_Registration]] - Tool collection across middleware

'''Implementation Details:'''
* [[langchain-ai_langchain_AgentMiddleware_base]] - Middleware base class
* [[langchain-ai_langchain_middleware_hooks]] - Individual hook implementations

'''Workflows:'''
* [[langchain-ai_langchain_Middleware_Composition]] - Complete workflow

[[Category:Principles]]
[[Category:Middleware Composition]]
[[Category:Execution Order]]
