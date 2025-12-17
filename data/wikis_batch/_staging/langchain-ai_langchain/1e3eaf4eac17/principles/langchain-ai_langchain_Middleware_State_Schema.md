= Middleware_State_Schema =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/middleware/types.py:L337
|domains=State Management, Schema Extension, Type Systems
|last_updated=2025-12-17
}}

== Overview ==

'''Middleware_State_Schema''' defines the mechanism by which middleware can extend the base agent state with custom fields, enabling middleware to store and access domain-specific data throughout the agent execution lifecycle. This principle ensures type-safe state extensions while maintaining backward compatibility with the core AgentState schema.

== Description ==

The principle centers on the `state_schema` class attribute on AgentMiddleware, which allows middleware to declare a custom TypedDict schema that extends AgentState:

'''Core Concepts:'''

* '''Schema Declaration''': Middleware declares `state_schema` as a class attribute pointing to a TypedDict subclass of AgentState
* '''Schema Merging''': The agent factory merges all middleware state schemas into a single resolved schema at graph compilation time
* '''Field Annotations''': Special annotations control schema visibility and persistence:
** `OmitFromSchema(input=True, output=False)` - Omit from input schema only
** `OmitFromSchema(input=False, output=True)` - Omit from output schema only
** `PrivateStateAttr` - Omit from both input and output (internal only)
** `EphemeralValue` - Value not persisted across invocations

* '''Type Safety''': TypedDict provides IDE autocomplete and type checking for custom state fields
* '''Reducer Functions''': State fields can use custom reducers via Annotated metadata (e.g., `add_messages` for message lists)

'''Standard AgentState Fields:'''

<source lang="python">
class AgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
</source>

'''Extension Pattern:'''

Middleware extends AgentState by creating a new TypedDict that inherits from it and adds custom fields:

<source lang="python">
class CustomState(AgentState):
    session_id: str  # Regular field
    user_context: Annotated[dict, PrivateStateAttr]  # Internal only
    cached_data: NotRequired[Annotated[dict, OmitFromOutput]]  # Input only
</source>

'''Schema Resolution Process:'''

1. Collect all `state_schema` attributes from middleware instances
2. Extract field type hints from each schema (including Annotated metadata)
3. Filter fields based on `OmitFromSchema` annotations for input/output schemas
4. Merge all fields into unified schemas: full, input-only, output-only
5. Create TypedDict classes dynamically for each schema variant
6. Apply schemas to StateGraph during compilation

== Usage ==

The Middleware_State_Schema principle is applied when:

* Middleware needs to persist data across multiple hook invocations
* Middleware requires domain-specific state fields (e.g., authentication tokens, session metadata)
* Type safety and IDE support are important for middleware state access
* Different middleware instances need isolated state namespaces
* State fields should be hidden from input/output schemas

'''Use Cases:'''

* '''Session Management''': Store session IDs, user identifiers, timestamps
* '''Authentication''': Cache tokens, user roles, permissions
* '''Caching''': Store intermediate results, cache keys, hit rates
* '''Metrics''': Track counters, timers, performance data
* '''Context''': Maintain conversation context, user preferences, settings
* '''Temporary Data''': Store data needed across hooks but not persisted

'''Field Annotation Selection:'''

* Use '''no annotation''' for:
** Fields that should be in both input and output schemas
** Standard state data that users provide and receive

* Use '''OmitFromInput''' for:
** Fields computed by middleware (not provided by users)
** Derived state that only appears in output
** Example: `structured_response`, computed metrics

* Use '''OmitFromOutput''' for:
** Fields only needed for initial configuration
** Temporary data consumed during execution
** Example: Initial cache seeds, one-time configs

* Use '''PrivateStateAttr''' for:
** Internal middleware bookkeeping
** State that should never be exposed to users
** Example: Internal counters, temporary calculations

* Use '''EphemeralValue''' for:
** Flow control fields (like `jump_to`)
** State that resets between invocations
** Values that shouldn't persist in checkpoints

== Theoretical Basis ==

The Middleware_State_Schema principle draws from several design patterns and type system concepts:

'''1. Open/Closed Principle (SOLID)'''

State schema extension allows the agent state to be open for extension (via middleware schemas) but closed for modification (base AgentState remains unchanged).

'''2. Decorator Pattern (Structural)'''

Each middleware schema decorates the base AgentState with additional fields, similar to how decorators add behavior in the structural pattern.

'''3. Mixins (Object-Oriented Programming)'''

The schema merging process resembles multiple inheritance or mixin composition, where multiple trait-like schemas combine into a single unified interface.

'''4. TypedDict Protocol (Python)'''

Leverages Python's TypedDict for structural typing:
* Provides static type checking without runtime overhead
* Enables IDE autocomplete and type inference
* Supports optional fields via `NotRequired`
* Allows metadata via `Annotated`

'''5. Schema Composition'''

Similar to GraphQL schema stitching or OpenAPI schema composition, where multiple schema fragments merge into a cohesive whole.

'''6. Visibility Modifiers'''

The `OmitFromSchema` annotations implement visibility modifiers (public/private/internal) at the schema level rather than the object level.

'''Design Principles:'''

* '''Composability''': Multiple middleware schemas combine without conflicts (assuming unique field names)
* '''Type Safety''': TypedDict provides compile-time checking and IDE support
* '''Isolation''': Middleware state is logically isolated through naming conventions
* '''Flexibility''': Annotations provide fine-grained control over field visibility
* '''Performance''': TypedDict has zero runtime overhead compared to dataclasses

'''Schema Merging Semantics:'''

* Fields with the same name must have compatible types across all middleware
* Last schema in middleware list "wins" if field types differ (implementation-dependent)
* Annotations from all schemas are preserved and merged
* Reducers apply to fields with matching names across schemas

== Related Pages ==

'''Implementation:'''
* [[langchain-ai_langchain_state_schema_extension]] - Schema merging implementation

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Base middleware abstraction
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Hook state access patterns
* [[langchain-ai_langchain_Middleware_Tool_Registration]] - Tool-related state

'''Implementation Details:'''
* [[langchain-ai_langchain_AgentMiddleware_base]] - State schema attribute
* [[langchain-ai_langchain_middleware_hooks]] - State access in hooks

'''Workflows:'''
* [[langchain-ai_langchain_Middleware_Composition]] - Complete workflow

[[Category:Principles]]
[[Category:State Management]]
[[Category:Type Systems]]
