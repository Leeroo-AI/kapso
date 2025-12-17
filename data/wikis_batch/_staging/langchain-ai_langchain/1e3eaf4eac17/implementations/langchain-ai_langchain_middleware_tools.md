= middleware_tools =

{{Metadata
|sources=libs/langchain_v1/langchain/agents/factory.py:L730, L765-792
|domains=Tool Systems, Tool Registration, Middleware Runtime
|last_updated=2025-12-17
}}

== Overview ==

'''middleware_tools''' implements the tool collection and registration mechanism that gathers tools from middleware instances and integrates them into the agent's ToolNode. This implementation ensures middleware-provided tools are available for model binding and execution alongside user-provided tools.

== Description ==

The implementation occurs in the `create_agent` function and involves several steps:

'''1. Tool Collection (Line 730)'''

<source lang="python">
middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]
</source>

* Iterates through all middleware instances
* Extracts the `tools` attribute from each (defaults to empty list if not present)
* Flattens into a single list of middleware tools

'''2. Tool Categorization (Lines 767-768)'''

<source lang="python">
built_in_tools = [t for t in tools if isinstance(t, dict)]
regular_tools = [t for t in tools if not isinstance(t, dict)]
</source>

* Separates user-provided tools into two categories:
** Built-in provider tools (dict format) - executed server-side by model provider
** Regular tools (BaseTool or callable) - executed client-side by ToolNode

'''3. Client-Side Tool Aggregation (Line 771)'''

<source lang="python">
available_tools = middleware_tools + regular_tools
</source>

* Combines middleware tools with regular user tools
* These require client-side execution (must be in ToolNode)
* Excludes built-in provider tools (handled differently)

'''4. ToolNode Creation (Lines 774-782)'''

<source lang="python">
tool_node = (
    ToolNode(
        tools=available_tools,
        wrap_tool_call=wrap_tool_call_wrapper,
        awrap_tool_call=awrap_tool_call_wrapper,
    )
    if available_tools
    else None
)
</source>

* Creates ToolNode only if there are client-side tools
* Passes combined tool list (middleware + user tools)
* Includes tool call wrappers from middleware
* ToolNode converts callables to BaseTool instances

'''5. Default Tools for ModelRequest (Lines 788-791)'''

<source lang="python">
if tool_node:
    default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
else:
    default_tools = list(built_in_tools)
</source>

* Creates default tool list for initial ModelRequest
* Uses converted BaseTool instances from ToolNode
* Includes built-in provider tools
* These tools can be modified by middleware before model call

== Code Reference ==

'''Complete Tool Registration Flow:'''

<source lang="python">
# Lines 730-792 in factory.py

# 1. Collect middleware tools
middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

# ... (intervening code for wrap_tool_call setup)

# 2. Extract built-in provider tools (dict format) and regular tools
built_in_tools = [t for t in tools if isinstance(t, dict)]
regular_tools = [t for t in tools if not isinstance(t, dict)]

# 3. Combine middleware tools with regular tools
available_tools = middleware_tools + regular_tools

# 4. Create ToolNode with all client-side tools
tool_node = (
    ToolNode(
        tools=available_tools,
        wrap_tool_call=wrap_tool_call_wrapper,
        awrap_tool_call=awrap_tool_call_wrapper,
    )
    if available_tools
    else None
)

# 5. Setup default tools for ModelRequest
if tool_node:
    default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
else:
    default_tools = list(built_in_tools)
</source>

'''Tool Validation (Lines 996-1019):'''

<source lang="python">
# Validate client-side tools exist in ToolNode
available_tools_by_name = {}
if tool_node:
    available_tools_by_name = tool_node.tools_by_name.copy()

# Check if any requested tools are unknown
unknown_tool_names = []
for t in request.tools:
    # Only validate BaseTool instances (skip built-in dict tools)
    if isinstance(t, dict):
        continue
    if isinstance(t, BaseTool) and t.name not in available_tools_by_name:
        unknown_tool_names.append(t.name)

if unknown_tool_names:
    available_tool_names = sorted(available_tools_by_name.keys())
    msg = (
        f"Middleware returned unknown tool names: {unknown_tool_names}\n\n"
        f"Available client-side tools: {available_tool_names}\n\n"
        "To fix this issue:\n"
        "1. Ensure the tools are passed to create_agent() via the 'tools' parameter\n"
        "2. If using custom middleware with tools, ensure they're registered via middleware.tools attribute\n"
        "3. Verify that tool names in ModelRequest.tools match the actual tool.name values\n"
    )
    raise ValueError(msg)
</source>

== I/O Contract ==

'''Tool Collection:'''

'''Input:'''
* Middleware instances with `tools` class attribute
* User-provided tools list from `create_agent(tools=...)`

'''Output:'''
* `middleware_tools: list[BaseTool]` - All tools from all middleware
* `available_tools: list[BaseTool | Callable]` - Combined middleware + user tools
* `tool_node: ToolNode | None` - ToolNode instance if tools exist

'''Processing:'''
1. Extract `tools` attribute from each middleware
2. Flatten into single list
3. Separate user tools by type (built-in vs. regular)
4. Combine middleware + regular tools
5. Create ToolNode with combined list
6. Build default_tools for ModelRequest

'''Tool Availability:'''

* '''Client-side tools''': Middleware tools + regular user tools
** Executed by ToolNode
** Available in `tool_node.tools_by_name`
** Can be wrapped by `wrap_tool_call` hooks

* '''Built-in provider tools''': User-provided dict tools
** Passed to model as dict objects
** Executed server-side by model provider
** Not in ToolNode
** Cannot be wrapped by middleware

'''Validation Rules:'''

* Tool names must be unique across all tools
* Middleware-returned tools (in ModelRequest) must exist in ToolNode
* Built-in tools can be dynamically added without validation

== Usage Examples ==

'''Example 1: Basic Tool Registration'''

<source lang="python">
from langchain_core.tools import tool
from langchain.agents.middleware.types import AgentMiddleware

@tool
def custom_search(query: str) -> str:
    """Search custom knowledge base."""
    return search_kb(query)

@tool
def custom_summarize(text: str) -> str:
    """Summarize long text."""
    return summarize(text)

class KnowledgeBaseMiddleware(AgentMiddleware):
    tools = [custom_search, custom_summarize]

    def wrap_tool_call(self, request, handler):
        """Log all KB tool calls."""
        if request.tool_call["name"] in ["custom_search", "custom_summarize"]:
            log_tool_call(request.tool_call)
        return handler(request)

# Usage
agent = create_agent(
    model="openai:gpt-4",
    middleware=[KnowledgeBaseMiddleware()]
)
# Agent now has access to custom_search and custom_summarize tools
</source>

'''Example 2: Tool and Hook Integration'''

<source lang="python">
from langchain_core.tools import tool
from langchain.agents.middleware.types import AgentMiddleware, ToolMessage

@tool
def query_database(query: str) -> str:
    """Execute SQL query on database."""
    return execute_query(query)

class DatabaseMiddleware(AgentMiddleware):
    tools = [query_database]

    def before_agent(self, state, runtime):
        """Initialize database connection."""
        return {"db_connection": create_connection()}

    def wrap_tool_call(self, request, handler):
        """Validate and sandbox database queries."""
        if request.tool_call["name"] == "query_database":
            query = request.tool_call["args"]["query"]

            # Validate query is read-only
            if not is_read_only(query):
                return ToolMessage(
                    content="Error: Only SELECT queries allowed",
                    tool_call_id=request.tool_call["id"],
                    status="error"
                )

        return handler(request)

    def after_agent(self, state, runtime):
        """Close database connection."""
        if conn := state.get("db_connection"):
            conn.close()
        return None
</source>

'''Example 3: Conditional Tool Registration'''

<source lang="python">
from langchain_core.tools import tool
from langchain.agents.middleware.types import AgentMiddleware

@tool
def admin_delete_user(user_id: str) -> str:
    """Delete a user account (admin only)."""
    return delete_user(user_id)

@tool
def admin_view_logs(log_level: str) -> str:
    """View system logs (admin only)."""
    return get_logs(log_level)

class AdminMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        # Tools set in __init__ based on configuration
        self.tools = [admin_delete_user, admin_view_logs]

    def before_agent(self, state, runtime):
        """Verify admin privileges."""
        user = runtime.context.get("user")
        if not is_admin(user):
            raise PermissionError("Admin middleware requires admin user")
        return None

# Only add AdminMiddleware if user is admin
middleware = []
if user.is_admin:
    middleware.append(AdminMiddleware())

agent = create_agent(model="openai:gpt-4", middleware=middleware)
</source>

'''Example 4: Tool Wrapper Pattern'''

<source lang="python">
from langchain_core.tools import tool, BaseTool
from langchain.agents.middleware.types import AgentMiddleware

# Original expensive tool
@tool
def expensive_api_call(query: str) -> str:
    """Call expensive external API."""
    return call_external_api(query)

# Cached wrapper tool
@tool
def cached_api_call(query: str) -> str:
    """Cached version of expensive API call."""
    # This is a wrapper that delegates to original
    if cached := get_from_cache(query):
        return cached
    result = call_external_api(query)
    save_to_cache(query, result)
    return result

class CachingMiddleware(AgentMiddleware):
    # Provide cached version instead of original
    tools = [cached_api_call]

# User provides original tool
agent = create_agent(
    model="openai:gpt-4",
    tools=[expensive_api_call],
    middleware=[CachingMiddleware()]
)
# Agent has both expensive_api_call and cached_api_call available
</source>

'''Example 5: Dynamic Tool Generation'''

<source lang="python">
from langchain_core.tools import tool, StructuredTool
from langchain.agents.middleware.types import AgentMiddleware

class PluginMiddleware(AgentMiddleware):
    def __init__(self, plugin_config: dict):
        super().__init__()
        # Generate tools dynamically based on configuration
        self.tools = self._generate_tools(plugin_config)

    def _generate_tools(self, config: dict) -> list[BaseTool]:
        """Generate tools based on plugin configuration."""
        tools = []

        for endpoint in config.get("endpoints", []):
            # Create a tool for each endpoint
            tool_func = self._create_endpoint_tool(endpoint)
            tool_obj = StructuredTool.from_function(
                func=tool_func,
                name=f"api_{endpoint['name']}",
                description=endpoint.get("description", "API endpoint")
            )
            tools.append(tool_obj)

        return tools

    def _create_endpoint_tool(self, endpoint: dict):
        """Create tool function for endpoint."""
        def call_endpoint(**kwargs) -> str:
            return call_api(endpoint["url"], kwargs)
        return call_endpoint

# Usage
plugin_config = {
    "endpoints": [
        {"name": "get_user", "url": "/api/users", "description": "Get user info"},
        {"name": "create_order", "url": "/api/orders", "description": "Create order"},
    ]
}

agent = create_agent(
    model="openai:gpt-4",
    middleware=[PluginMiddleware(plugin_config)]
)
# Agent has api_get_user and api_create_order tools
</source>

'''Example 6: Tool Namespace Management'''

<source lang="python">
from langchain_core.tools import tool
from langchain.agents.middleware.types import AgentMiddleware

# Middleware 1: Database tools
@tool
def db_query(query: str) -> str:
    """Query database."""
    return query_db(query)

@tool
def db_insert(data: dict) -> str:
    """Insert into database."""
    return insert_db(data)

class DatabaseMiddleware(AgentMiddleware):
    tools = [db_query, db_insert]

# Middleware 2: File tools
@tool
def file_read(path: str) -> str:
    """Read file."""
    return read_file(path)

@tool
def file_write(path: str, content: str) -> str:
    """Write file."""
    return write_file(path, content)

class FileMiddleware(AgentMiddleware):
    tools = [file_read, file_write]

# Both middleware can coexist without name conflicts
# Tool names are prefixed with domain (db_, file_)
agent = create_agent(
    model="openai:gpt-4",
    middleware=[DatabaseMiddleware(), FileMiddleware()]
)

# Agent has access to: db_query, db_insert, file_read, file_write
</source>

== Related Pages ==

'''Principle:'''
* [[langchain-ai_langchain_Middleware_Tool_Registration]] - Tool registration concept

'''Related Implementations:'''
* [[langchain-ai_langchain_AgentMiddleware_base]] - Tools attribute
* [[langchain-ai_langchain_middleware_hooks]] - wrap_tool_call implementation
* [[langchain-ai_langchain_chain_handlers]] - Tool wrapper composition

'''Related Principles:'''
* [[langchain-ai_langchain_Middleware_Definition]] - Middleware abstraction
* [[langchain-ai_langchain_Middleware_Lifecycle_Hooks]] - Tool-related hooks

[[Category:Implementations]]
[[Category:Tool Systems]]
[[Category:Agent Capabilities]]
