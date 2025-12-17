= state_schema_extension =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/factory.py:L283-326, L852-859
|domains=State Management, Schema Merging, Type System Runtime
|last_updated=2025-12-17
}}

== Overview ==

'''state_schema_extension''' implements the runtime schema merging logic that combines middleware state schemas into unified TypedDict schemas for the agent graph. This implementation handles schema resolution, field filtering based on visibility annotations, and dynamic TypedDict creation.

== Description ==

The implementation consists of three key functions in the agent factory:

'''1. _resolve_schema'''

Main function that merges multiple TypedDict schemas while respecting visibility annotations:
* Takes a set of schema types and a schema name
* Optionally filters fields based on `omit_flag` ('input' or 'output')
* Extracts type hints from all schemas
* Checks each field for `OmitFromSchema` annotations
* Merges non-omitted fields into a new TypedDict
* Returns dynamically created TypedDict class

'''2. _extract_metadata'''

Helper function that extracts metadata from Annotated type hints:
* Handles `Required[Annotated[...]]` and `NotRequired[Annotated[...]]` wrappers
* Handles direct `Annotated[...]` types
* Returns list of metadata objects from the annotation
* Used to find `OmitFromSchema` annotations

'''3. Schema Collection and Application'''

The `create_agent` function orchestrates schema resolution:
* Collects `state_schema` attributes from all middleware
* Adds user-provided `state_schema` parameter or base `AgentState`
* Calls `_resolve_schema` three times to create:
** Full internal schema (no filtering)
** Input schema (omits fields with `input=True`)
** Output schema (omits fields with `output=True`)
* Applies schemas to StateGraph constructor

== Code Reference ==

'''Schema Resolution Implementation:'''

<source lang="python">
# _resolve_schema function (factory.py lines 283-311)
def _resolve_schema(schemas: set[type], schema_name: str, omit_flag: str | None = None) -> type:
    """Resolve schema by merging schemas and optionally respecting `OmitFromSchema` annotations.

    Args:
        schemas: List of schema types to merge
        schema_name: Name for the generated `TypedDict`
        omit_flag: If specified, omit fields with this flag set (`'input'` or
            `'output'`)
    """
    all_annotations = {}

    for schema in schemas:
        hints = get_type_hints(schema, include_extras=True)

        for field_name, field_type in hints.items():
            should_omit = False

            if omit_flag:
                # Check for omission in the annotation metadata
                metadata = _extract_metadata(field_type)
                for meta in metadata:
                    if isinstance(meta, OmitFromSchema) and getattr(meta, omit_flag) is True:
                        should_omit = True
                        break

            if not should_omit:
                all_annotations[field_name] = field_type

    return TypedDict(schema_name, all_annotations)  # type: ignore[operator]
</source>

'''Metadata Extraction:'''

<source lang="python">
# _extract_metadata function (factory.py lines 314-326)
def _extract_metadata(type_: type) -> list:
    """Extract metadata from a field type, handling Required/NotRequired and Annotated wrappers."""
    # Handle Required[Annotated[...]] or NotRequired[Annotated[...]]
    if get_origin(type_) in (Required, NotRequired):
        inner_type = get_args(type_)[0]
        if get_origin(inner_type) is Annotated:
            return list(get_args(inner_type)[1:])

    # Handle direct Annotated[...]
    elif get_origin(type_) is Annotated:
        return list(get_args(type_)[1:])

    return []
</source>

'''Schema Collection and Application:'''

<source lang="python">
# Schema collection (factory.py lines 852-859)
state_schemas: set[type] = {m.state_schema for m in middleware}
# Use provided state_schema if available, otherwise use base AgentState
base_state = state_schema if state_schema is not None else AgentState
state_schemas.add(base_state)

resolved_state_schema = _resolve_schema(state_schemas, "StateSchema", None)
input_schema = _resolve_schema(state_schemas, "InputSchema", "input")
output_schema = _resolve_schema(state_schemas, "OutputSchema", "output")

# StateGraph creation (factory.py lines 862-869)
graph: StateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]
graph = StateGraph(
    state_schema=resolved_state_schema,
    input_schema=input_schema,
    output_schema=output_schema,
    context_schema=context_schema,
)
</source>

== I/O Contract ==

'''_resolve_schema Function:'''

'''Input:'''
* `schemas: set[type]` - Set of TypedDict classes to merge
* `schema_name: str` - Name for the generated TypedDict
* `omit_flag: str | None` - Optional filter: 'input' or 'output'

'''Output:'''
* `type` - Dynamically created TypedDict class with merged fields

'''Processing:'''
1. For each schema in the set:
   - Extract type hints including Annotated metadata
   - For each field:
     * Extract metadata using `_extract_metadata`
     * Check if field has `OmitFromSchema` with matching `omit_flag`
     * If not omitted, add to `all_annotations` dict
2. Create new TypedDict with merged annotations
3. Return new TypedDict class

'''_extract_metadata Function:'''

'''Input:'''
* `type_: type` - Field type annotation (may be wrapped in Required/NotRequired/Annotated)

'''Output:'''
* `list` - List of metadata objects from Annotated type

'''Processing:'''
1. Check if type is Required or NotRequired
   - If yes, extract inner type
   - Check if inner type is Annotated
2. If type is directly Annotated
   - Extract metadata items
3. Return metadata list (empty if not annotated)

'''Schema Application:'''

'''Input:'''
* Middleware instances with `state_schema` attributes
* Optional user-provided `state_schema` parameter
* Base `AgentState` class

'''Output:'''
* Three resolved schemas: full (internal), input, output
* StateGraph configured with these schemas

== Usage Examples ==

'''Example 1: Basic State Extension'''

<source lang="python">
from typing import Annotated, TypedDict, NotRequired
from langchain.agents.middleware.types import AgentMiddleware, AgentState

class SessionState(AgentState):
    session_id: str
    user_id: str
    created_at: str

class SessionMiddleware(AgentMiddleware):
    state_schema = SessionState

    def before_agent(self, state: SessionState, runtime):
        # Type-safe access to custom fields
        return {
            "session_id": generate_id(),
            "user_id": runtime.context.get("user_id", "anonymous"),
            "created_at": datetime.now().isoformat()
        }

# Usage
agent = create_agent(
    model="openai:gpt-4",
    middleware=[SessionMiddleware()]
)
</source>

'''Example 2: Private Internal State'''

<source lang="python">
from langchain.agents.middleware.types import PrivateStateAttr

class CacheState(AgentState):
    # Internal cache not exposed in input/output
    _cache: Annotated[dict, PrivateStateAttr] = {}
    # Cache hit rate exposed in output only
    cache_hit_rate: Annotated[float, OmitFromInput]

class CacheMiddleware(AgentMiddleware):
    state_schema = CacheState

    def before_agent(self, state: CacheState, runtime):
        return {"_cache": {}, "cache_hit_rate": 0.0}

    def wrap_model_call(self, request, handler):
        cache_key = hash_request(request)
        cache = request.state.get("_cache", {})

        if cache_key in cache:
            # Update hit rate
            hits = cache.get("_hits", 0) + 1
            total = cache.get("_total", 0) + 1
            return ModelResponse(
                result=[cache[cache_key]],
                structured_response=None
            )

        response = handler(request)
        cache[cache_key] = response.result[0]
        cache["_total"] = cache.get("_total", 0) + 1

        return response
</source>

'''Example 3: Multiple Middleware with Merged State'''

<source lang="python">
# Middleware 1: Authentication
class AuthState(AgentState):
    user_token: Annotated[str, PrivateStateAttr]
    user_role: str

class AuthMiddleware(AgentMiddleware):
    state_schema = AuthState

    def before_agent(self, state: AuthState, runtime):
        token = runtime.context.get("auth_token")
        role = verify_token(token)
        return {"user_token": token, "user_role": role}

# Middleware 2: Metrics
class MetricsState(AgentState):
    request_count: Annotated[int, PrivateStateAttr] = 0
    total_latency: Annotated[float, OmitFromInput] = 0.0

class MetricsMiddleware(AgentMiddleware):
    state_schema = MetricsState

    def before_agent(self, state: MetricsState, runtime):
        return {"request_count": 0, "total_latency": 0.0}

    def wrap_model_call(self, request, handler):
        start = time.time()
        response = handler(request)
        latency = time.time() - start

        # Update metrics
        count = request.state.get("request_count", 0) + 1
        total = request.state.get("total_latency", 0.0) + latency

        return response

# Usage - schemas are automatically merged
agent = create_agent(
    model="openai:gpt-4",
    middleware=[AuthMiddleware(), MetricsMiddleware()]
)

# Effective state schema includes fields from both middleware:
# - messages (from AgentState)
# - user_role (from AuthState, public)
# - user_token (from AuthState, private)
# - request_count (from MetricsState, private)
# - total_latency (from MetricsState, output only)
</source>

'''Example 4: User-Provided Base State Schema'''

<source lang="python">
# User defines custom base state
class MyAppState(AgentState):
    app_version: str
    feature_flags: dict

# Middleware extends it further
class LoggingState(MyAppState):
    log_entries: Annotated[list[str], PrivateStateAttr] = []

class LoggingMiddleware(AgentMiddleware):
    state_schema = LoggingState

    def before_model(self, state: LoggingState, runtime):
        logs = state.get("log_entries", [])
        logs.append(f"Before model: {len(state['messages'])} messages")
        return {"log_entries": logs}

# Usage - user state is the base
agent = create_agent(
    model="openai:gpt-4",
    middleware=[LoggingMiddleware()],
    state_schema=MyAppState  # User's base state
)

# Initial invocation must provide MyAppState fields
response = agent.invoke({
    "messages": [{"role": "user", "content": "Hello"}],
    "app_version": "1.0.0",
    "feature_flags": {"new_ui": True}
})
</source>

'''Example 5: Ephemeral State Fields'''

<source lang="python">
from langgraph.channels.ephemeral_value import EphemeralValue

class WorkflowState(AgentState):
    # Ephemeral field resets between invocations
    current_step: Annotated[str | None, EphemeralValue, PrivateStateAttr]
    # Persistent field maintained across invocations (with checkpointing)
    workflow_history: list[str] = []

class WorkflowMiddleware(AgentMiddleware):
    state_schema = WorkflowState

    def before_model(self, state: WorkflowState, runtime):
        # current_step is always None at start of new invocation
        history = state.get("workflow_history", [])
        history.append("model_step")
        return {
            "current_step": "calling_model",
            "workflow_history": history
        }

    def after_model(self, state: WorkflowState, runtime):
        # current_step is available within same invocation
        return {"current_step": None}
</source>

'''Example 6: State Schema with Custom Reducers'''

<source lang="python">
from typing import Annotated

def merge_metadata(existing: dict, new: dict) -> dict:
    """Custom reducer that merges dictionaries."""
    return {**existing, **new}

class MetadataState(AgentState):
    # Custom reducer for metadata field
    metadata: Annotated[dict, merge_metadata] = {}

class MetadataMiddleware(AgentMiddleware):
    state_schema = MetadataState

    def before_agent(self, state: MetadataState, runtime):
        # Metadata fields are merged, not replaced
        return {"metadata": {"session_start": datetime.now().isoformat()}}

    def after_agent(self, state: MetadataState, runtime):
        # Additional metadata is merged with existing
        return {"metadata": {"session_end": datetime.now().isoformat()}}

# Final state will have both session_start and session_end in metadata
</source>

== Related Pages ==

'''Principle:'''
* [[langchain-ai_langchain_Middleware_State_Schema]] - State extension concept

'''Related Implementations:'''
* [[langchain-ai_langchain_AgentMiddleware_base]] - State schema attribute
* [[langchain-ai_langchain_middleware_hooks]] - State access in hooks

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Middleware abstraction
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Hook patterns

[[Category:Implementations]]
[[Category:State Management]]
[[Category:Schema Systems]]
