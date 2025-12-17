= Middleware_Tool_Registration =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/middleware/types.py:L340-341
|domains=Tool Systems, Agent Capabilities, Middleware Extensions
|last_updated=2025-12-17
}}

== Overview ==

'''Middleware_Tool_Registration''' defines the mechanism by which middleware instances can contribute additional tools to the agent's capabilities without requiring those tools to be specified in the `create_agent` call. This principle enables middleware to package related functionality (tools + hooks) into cohesive, reusable components.

== Description ==

The principle centers on the `tools` class attribute on AgentMiddleware, which allows middleware to declare a list of tools that should be automatically registered with the agent:

'''Core Concepts:'''

* '''Tool Declaration''': Middleware declares `tools` as a class attribute containing a list of `BaseTool` instances
* '''Automatic Registration''': Tools from all middleware are automatically collected and added to the agent's tool set
* '''Tool-Hook Coupling''': Middleware can provide tools alongside hooks that interact with those tools
* '''Scoped Tools''': Tools are logically scoped to the middleware that provides them

'''Tool Collection Process:'''

1. Agent factory collects `tools` attribute from all middleware instances
2. Middleware tools are merged with user-provided tools from `create_agent(tools=...)`
3. All tools are passed to ToolNode for execution
4. Tools become available for model binding and invocation

'''Use Cases:'''

* '''Domain-Specific Tools''': Middleware provides specialized tools for its domain (e.g., database middleware provides query tools)
* '''Hook-Tool Integration''': Tools that are monitored/modified by middleware hooks
* '''Conditional Tools''': Tools that are only available when middleware is enabled
* '''Tool Wrappers''': Middleware provides wrapper tools that delegate to underlying implementations

== Usage ==

The Middleware_Tool_Registration principle is applied when:

* Middleware needs to extend agent capabilities with domain-specific operations
* Tools require accompanying hooks for proper operation (validation, logging, etc.)
* Creating reusable middleware packages with self-contained functionality
* Implementing plugin-style architecture where middleware adds features

'''Design Patterns:'''

'''1. Tool Provider Pattern'''

Middleware acts as a tool provider, encapsulating tool implementations alongside usage logic:

<source lang="python">
class DatabaseMiddleware(AgentMiddleware):
    tools = [query_tool, insert_tool, update_tool]

    def wrap_tool_call(self, request, handler):
        # Monitor and validate database operations
        if request.tool_call["name"] in ["query_tool", "insert_tool", "update_tool"]:
            validate_db_args(request.tool_call["args"])
        return handler(request)
</source>

'''2. Tool Wrapper Pattern'''

Middleware provides wrapper tools that enhance existing functionality:

<source lang="python">
class CachingMiddleware(AgentMiddleware):
    tools = [cached_search_tool]  # Wraps original search tool

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] == "cached_search_tool":
            # Check cache before calling wrapped tool
            if cached := get_cache(request.tool_call["args"]):
                return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
        return handler(request)
</source>

'''3. Tool Set Extension Pattern'''

Middleware adds complementary tools to existing agent capabilities:

<source lang="python">
class AdminMiddleware(AgentMiddleware):
    tools = [admin_panel_tool, user_management_tool]

    def before_agent(self, state, runtime):
        # Check if user has admin privileges
        if not is_admin(runtime.context.get("user")):
            # Could raise exception or remove admin tools
            pass
</source>

== Theoretical Basis ==

The Middleware_Tool_Registration principle draws from several software design patterns:

'''1. Plugin Architecture'''

Middleware tools implement a plugin system where:
* Plugins (middleware) can extend core functionality (agent capabilities)
* Plugins are self-contained units with their own tools and logic
* Plugins can be added/removed without modifying core agent code

'''2. Strategy Pattern'''

Different middleware instances provide different tool strategies:
* Each middleware brings its own tool implementations
* Tools can be swapped by changing middleware configuration
* Multiple strategies can coexist (multiple middleware)

'''3. Decorator Pattern (Behavioral)'''

Middleware decorates the agent with additional capabilities (tools):
* Base agent has core functionality
* Middleware layers add tools incrementally
* Each middleware layer is independent

'''4. Inversion of Control (IoC)'''

The agent doesn't know about specific tools at compile time:
* Middleware controls which tools are available
* Agent discovers tools at runtime during graph compilation
* Loose coupling between agent core and tool implementations

'''5. Cohesion Principle'''

Related functionality stays together:
* Tools and their management hooks are co-located in middleware
* High cohesion within middleware (tools + hooks for same domain)
* Loose coupling between middleware instances

'''Design Principles:'''

* '''Single Responsibility''': Each middleware is responsible for its domain's tools
* '''Open/Closed''': Agent is open for tool extension (via middleware) but closed for modification
* '''Dependency Injection''': Tools are injected into agent via middleware rather than hardcoded
* '''Composition over Inheritance''': Agent capabilities are composed from middleware rather than inherited

'''Namespace Considerations:'''

* Tool names must be unique across all middleware and user-provided tools
* Naming convention recommendation: prefix tool names with middleware domain
* Example: `db_query`, `db_insert` rather than generic `query`, `insert`

== Related Pages ==

'''Implementation:'''
* [[langchain-ai_langchain_middleware_tools]] - Tool collection and registration implementation

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Base middleware abstraction
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Tool call hooks (wrap_tool_call)
* [[langchain-ai_langchain_Middleware_State_Schema]] - Tool-related state storage

'''Implementation Details:'''
* [[langchain-ai_langchain_AgentMiddleware_base]] - Tools attribute definition
* [[langchain-ai_langchain_chain_handlers]] - Tool call wrapper composition

'''Workflows:'''
* [[langchain-ai_langchain_Middleware_Composition]] - Complete workflow

[[Category:Principles]]
[[Category:Tool Systems]]
[[Category:Agent Capabilities]]
