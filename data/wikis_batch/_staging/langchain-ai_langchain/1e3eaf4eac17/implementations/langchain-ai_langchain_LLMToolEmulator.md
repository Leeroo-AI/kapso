# LLMToolEmulator Implementation

## Metadata
- **Component**: `LLMToolEmulator`
- **Package**: `langchain.agents.middleware.tool_emulator`
- **File Path**: `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/tool_emulator.py`
- **Type**: Agent Middleware
- **Lines of Code**: 209
- **Related Classes**: None

## Overview

`LLMToolEmulator` is a testing middleware that replaces actual tool execution with LLM-generated responses. It allows selective emulation of tools for testing purposes without executing real tool code, making it valuable for development, testing, and demonstrations where actual tool execution is impractical or undesirable.

### Purpose
The middleware addresses testing and development challenges by enabling:
- Testing agent behavior without external dependencies
- Demonstrating agents without API keys or live services
- Rapid prototyping without implementing all tools
- Safely testing agents that use dangerous tools (file system, network, etc.)

### Key Features
- **Selective emulation**: Choose which tools to emulate (or emulate all by default)
- **LLM-powered responses**: Uses language model to generate realistic tool outputs
- **Configurable model**: Choose emulation model (defaults to Claude Sonnet)
- **Transparent substitution**: Seamlessly replaces tool execution
- **String or instance specification**: Specify tools by name or `BaseTool` instance

## Code Reference

### Main Class Definition

```python
class LLMToolEmulator(AgentMiddleware):
    """Emulates specified tools using an LLM instead of executing them.

    This middleware allows selective emulation of tools for testing purposes.

    By default (when `tools=None`), all tools are emulated. You can specify which
    tools to emulate by passing a list of tool names or `BaseTool` instances.

    Examples:
        !!! example "Emulate all tools (default behavior)"

            ```python
            from langchain.agents.middleware import LLMToolEmulator

            middleware = LLMToolEmulator()

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[get_weather, get_user_location, calculator],
                middleware=[middleware],
            )
            ```

        !!! example "Emulate specific tools by name"

            ```python
            middleware = LLMToolEmulator(tools=["get_weather", "get_user_location"])
            ```

        !!! example "Use a custom model for emulation"

            ```python
            middleware = LLMToolEmulator(
                tools=["get_weather"], model="anthropic:claude-sonnet-4-5-20250929"
            )
            ```

        !!! example "Emulate specific tools by passing tool instances"

            ```python
            middleware = LLMToolEmulator(tools=[get_weather, get_user_location])
            ```
    """

    def __init__(
        self,
        *,
        tools: list[str | BaseTool] | None = None,
        model: str | BaseChatModel | None = None,
    ) -> None:
        """Initialize the tool emulator.

        Args:
            tools: List of tool names (`str`) or `BaseTool` instances to emulate.

                If `None`, ALL tools will be emulated.

                If empty list, no tools will be emulated.
            model: Model to use for emulation.

                Defaults to `'anthropic:claude-sonnet-4-5-20250929'`.

                Can be a model identifier string or `BaseChatModel` instance.
        """
        super().__init__()

        # Extract tool names from tools
        # None means emulate all tools
        self.emulate_all = tools is None
        self.tools_to_emulate: set[str] = set()

        if not self.emulate_all and tools is not None:
            for tool in tools:
                if isinstance(tool, str):
                    self.tools_to_emulate.add(tool)
                else:
                    # Assume BaseTool with .name attribute
                    self.tools_to_emulate.add(tool.name)

        # Initialize emulator model
        if model is None:
            self.model = init_chat_model("anthropic:claude-sonnet-4-5-20250929", temperature=1)
        elif isinstance(model, BaseChatModel):
            self.model = model
        else:
            self.model = init_chat_model(model, temperature=1)
```

### Sync Tool Call Wrapping

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Emulate tool execution using LLM if tool should be emulated.

    Args:
        request: Tool call request to potentially emulate.
        handler: Callback to execute the tool (can be called multiple times).

    Returns:
        ToolMessage with emulated response if tool should be emulated,
            otherwise calls handler for normal execution.
    """
    tool_name = request.tool_call["name"]

    # Check if this tool should be emulated
    should_emulate = self.emulate_all or tool_name in self.tools_to_emulate

    if not should_emulate:
        # Let it execute normally by calling the handler
        return handler(request)

    # Extract tool information for emulation
    tool_args = request.tool_call["args"]
    tool_description = request.tool.description if request.tool else "No description available"

    # Build prompt for emulator LLM
    prompt = (
        f"You are emulating a tool call for testing purposes.\n\n"
        f"Tool: {tool_name}\n"
        f"Description: {tool_description}\n"
        f"Arguments: {tool_args}\n\n"
        f"Generate a realistic response that this tool would return "
        f"given these arguments.\n"
        f"Return ONLY the tool's output, no explanation or preamble. "
        f"Introduce variation into your responses."
    )

    # Get emulated response from LLM
    response = self.model.invoke([HumanMessage(prompt)])

    # Short-circuit: return emulated result without executing real tool
    return ToolMessage(
        content=response.content,
        tool_call_id=request.tool_call["id"],
        name=tool_name,
    )
```

### Async Tool Call Wrapping

```python
async def awrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    """Async version of `wrap_tool_call`.

    Emulate tool execution using LLM if tool should be emulated.

    Args:
        request: Tool call request to potentially emulate.
        handler: Async callback to execute the tool (can be called multiple times).

    Returns:
        ToolMessage with emulated response if tool should be emulated,
            otherwise calls handler for normal execution.
    """
    tool_name = request.tool_call["name"]

    # Check if this tool should be emulated
    should_emulate = self.emulate_all or tool_name in self.tools_to_emulate

    if not should_emulate:
        # Let it execute normally by calling the handler
        return await handler(request)

    # Extract tool information for emulation
    tool_args = request.tool_call["args"]
    tool_description = request.tool.description if request.tool else "No description available"

    # Build prompt for emulator LLM
    prompt = (
        f"You are emulating a tool call for testing purposes.\n\n"
        f"Tool: {tool_name}\n"
        f"Description: {tool_description}\n"
        f"Arguments: {tool_args}\n\n"
        f"Generate a realistic response that this tool would return "
        f"given these arguments.\n"
        f"Return ONLY the tool's output, no explanation or preamble. "
        f"Introduce variation into your responses."
    )

    # Get emulated response from LLM (using async invoke)
    response = await self.model.ainvoke([HumanMessage(prompt)])

    # Short-circuit: return emulated result without executing real tool
    return ToolMessage(
        content=response.content,
        tool_call_id=request.tool_call["id"],
        name=tool_name,
    )
```

## I/O Contract

### Input
- **ToolCallRequest**: Contains tool call details (name, arguments, tool instance)
- **Configuration**:
  - `tools`: List of tools to emulate (None = all, [] = none)
  - `model`: Model for generating emulated responses

### Output
- **ToolMessage**: Contains emulated response with proper tool_call_id
- **Command**: Pass-through if handler returns Command
- **Real Execution**: Falls through to handler if tool not in emulation list

### Emulation Decision Flow
1. Extract tool name from request
2. Check if `emulate_all=True` OR tool name in `tools_to_emulate` set
3. If yes: Generate emulated response via LLM
4. If no: Call handler for normal execution

### Prompt Structure
The emulation prompt includes:
- Tool name
- Tool description (from tool definition)
- Arguments passed to tool
- Instructions for realistic output
- Request for variation in responses

## Usage Examples

### Emulate All Tools (Default)

```python
from langchain.agents.middleware.tool_emulator import LLMToolEmulator
from langchain.agents import create_agent

# Emulate all tools by default
emulator = LLMToolEmulator()

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather, search_web, send_email],
    middleware=[emulator],
)

# All tool calls will be emulated - no real execution
result = await agent.invoke({
    "messages": [HumanMessage("What's the weather in Paris?")]
})
# get_weather not actually called - LLM generates realistic weather data
```

### Emulate Specific Tools by Name

```python
# Only emulate dangerous or expensive tools
emulator = LLMToolEmulator(
    tools=["send_email", "delete_file", "charge_credit_card"]
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, send_email, delete_file, charge_credit_card],
    middleware=[emulator],
)

# search_web executes normally
# send_email, delete_file, charge_credit_card are emulated
```

### Emulate Specific Tools by Instance

```python
from langchain.tools import BaseTool

# Define tools
weather_tool = create_weather_tool()
email_tool = create_email_tool()
calculator_tool = create_calculator_tool()

# Emulate only weather and email
emulator = LLMToolEmulator(
    tools=[weather_tool, email_tool]
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[weather_tool, email_tool, calculator_tool],
    middleware=[emulator],
)

# calculator_tool executes normally, others emulated
```

### Custom Emulation Model

```python
# Use cheaper/faster model for emulation
emulator = LLMToolEmulator(
    tools=["get_weather", "get_stock_price"],
    model="openai:gpt-4o-mini"  # Faster, cheaper model
)

agent = create_agent(
    model="openai:gpt-4o",
    middleware=[emulator],
)
```

### Development Workflow

```python
# During development: emulate unimplemented tools
emulator = LLMToolEmulator(
    tools=["future_feature_tool", "not_yet_implemented"]
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[
        working_tool,           # Real execution
        future_feature_tool,    # Emulated - not implemented yet
        not_yet_implemented,    # Emulated - not implemented yet
    ],
    middleware=[emulator],
)

# Can test agent behavior before all tools are implemented
```

### Testing Without Dependencies

```python
# Test agent without API keys or external services
emulator = LLMToolEmulator()  # Emulate all

agent = create_agent(
    model="openai:gpt-4o",
    tools=[
        stripe_charge_tool,      # Would need Stripe API key
        sendgrid_email_tool,     # Would need SendGrid API key
        aws_s3_upload_tool,      # Would need AWS credentials
    ],
    middleware=[emulator],
)

# Can test agent logic without any real credentials
result = await agent.invoke({
    "messages": [HumanMessage("Send invoice to customer")]
})
```

### Demonstration Mode

```python
# Safe demonstrations without side effects
emulator = LLMToolEmulator()

demo_agent = create_agent(
    model="openai:gpt-4o",
    tools=[delete_database, send_notifications, update_production],
    middleware=[emulator],
)

# Can safely demo agent in front of audience
# No risk of accidentally executing dangerous operations
```

### Selective Real Execution

```python
# Emulate most tools, execute only safe ones
emulator = LLMToolEmulator(
    tools=[]  # Empty list = emulate nothing (all real)
)

# Actually, better pattern:
safe_emulator = LLMToolEmulator(
    tools=[
        "dangerous_tool_1",
        "expensive_api_call",
        "slow_operation",
    ]
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[
        read_file,            # Real execution
        dangerous_tool_1,     # Emulated
        expensive_api_call,   # Emulated
        slow_operation,       # Emulated
    ],
    middleware=[safe_emulator],
)
```

### Pre-Initialized Model

```python
from langchain_anthropic import ChatAnthropic

# Use custom model configuration
custom_model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=1.0,
    max_tokens=500,
)

emulator = LLMToolEmulator(
    model=custom_model
)
```

## Implementation Details

### Tool Name Extraction

The middleware extracts tool names during initialization:
```python
for tool in tools:
    if isinstance(tool, str):
        self.tools_to_emulate.add(tool)
    else:
        # Assume BaseTool with .name attribute
        self.tools_to_emulate.add(tool.name)
```

Supports both:
- String tool names: `"get_weather"`
- Tool instances: Uses `.name` attribute

### Emulate All Flag

Special handling for `tools=None`:
```python
self.emulate_all = tools is None
```

This means:
- `tools=None`: Emulate all tools
- `tools=[]`: Emulate no tools (all real)
- `tools=["foo"]`: Emulate only "foo"

### Temperature Setting

Default model initialized with `temperature=1`:
```python
self.model = init_chat_model("anthropic:claude-sonnet-4-5-20250929", temperature=1)
```

Higher temperature increases variation in emulated responses.

### Prompt Engineering

The emulation prompt is carefully structured:
1. **Context**: "emulating a tool call for testing purposes"
2. **Tool Info**: Name, description, arguments
3. **Task**: "Generate a realistic response"
4. **Constraints**: "Return ONLY the tool's output, no explanation"
5. **Variation**: "Introduce variation into your responses"

This produces realistic, varied tool outputs.

### Short-Circuit Pattern

When emulating, the middleware returns immediately without calling handler:
```python
if should_emulate:
    response = self.model.invoke([HumanMessage(prompt)])
    return ToolMessage(...)  # Short-circuit
else:
    return handler(request)  # Normal execution
```

### Tool Metadata Preservation

The returned ToolMessage preserves:
- `tool_call_id`: From original request
- `name`: Tool name from request
- `content`: LLM-generated response

This ensures proper message history tracking.

### Model Initialization

Model can be specified as:
1. `None`: Uses default Claude Sonnet with temperature=1
2. String: Calls `init_chat_model(model, temperature=1)`
3. Instance: Uses provided model as-is

### Async and Sync Implementations

Both versions fully implemented:
- `wrap_tool_call`: Uses `model.invoke`
- `awrap_tool_call`: Uses `model.ainvoke`

Logic is duplicated for proper async support.

## Related Pages

### Core Middleware Infrastructure
- **langchain-ai_langchain_AgentMiddleware_class.md**: Base middleware interface
- **langchain-ai_langchain_middleware_hooks.txt**: Middleware hook system
- **langchain-ai_langchain_middleware_tools.txt**: Tool wrapping patterns

### Other V1 Middleware Implementations
- **langchain-ai_langchain_ContextEditingMiddleware.md**: Context window management
- **langchain-ai_langchain_ModelCallLimitMiddleware.md**: Model call quota enforcement
- **langchain-ai_langchain_ModelFallbackMiddleware.md**: Model failover on errors
- **langchain-ai_langchain_TodoListMiddleware.md**: Task tracking for agents

### Related Middleware
- **langchain-ai_langchain_ToolRetryMiddleware.md**: Retry logic for tools
- **langchain-ai_langchain_ToolCallLimitMiddleware.md**: Tool call limits

### Agent Creation
- **langchain-ai_langchain_create_agent.md**: Agent creation with middleware support

### Tool System
- **langchain-ai_langchain_BaseTool_creation.md**: Tool creation patterns

## Architecture Notes

### Design Philosophy
The middleware follows these principles:
- **Selective control**: Choose what to emulate
- **Transparent substitution**: Same interface as real execution
- **Realistic outputs**: LLM generates plausible responses
- **Safe testing**: No side effects from tool execution

### Use Case Categories

**Development**:
- Test agents before tools are implemented
- Prototype agent behavior quickly
- Develop without external dependencies

**Testing**:
- Unit test agent logic without tool side effects
- Integration test without API keys
- Test error handling without triggering errors

**Demonstration**:
- Show agent capabilities safely
- Demo without credentials or setup
- Present to audiences without risk

**Safety**:
- Test dangerous tools without risk
- Verify agent behavior with destructive operations
- Sandbox agent experimentation

### Comparison with Mocking

**LLMToolEmulator**:
- Generates dynamic, varied responses
- Responses contextual to arguments
- Realistic content based on tool description
- No need to pre-script responses

**Traditional Mocking**:
- Fixed responses per test
- Manual response scripting required
- Less realistic variation
- More predictable (pro and con)

Both approaches have value - LLM emulation for realistic behavior, mocking for deterministic tests.

### Performance Considerations
- Each emulated tool call makes LLM API call
- Emulation may be slower than real tool (or faster, depends on tool)
- Model initialization happens once at middleware creation
- Prompt construction is lightweight

### Cost Considerations
- Emulation has LLM API cost
- May be cheaper than real tool (e.g., expensive APIs)
- May be more expensive than real tool (e.g., local operations)
- Default Claude Sonnet is relatively expensive

### Realism Trade-offs

**More Realistic**:
- Uses tool description for context
- Varies responses (temperature=1)
- Contextual to arguments

**Less Realistic**:
- No domain-specific logic
- May hallucinate impossible data
- No validation of arguments
- No error conditions emulated

### Error Handling

The middleware does not emulate errors:
- Always returns successful ToolMessage
- No exception raising
- No validation failures
- Consider extending for error emulation

## Extension Points

### Error Emulation

```python
class ErrorEmulatingToolEmulator(LLMToolEmulator):
    def __init__(self, *args, error_rate=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_rate = error_rate

    def wrap_tool_call(self, request, handler):
        if should_emulate(request.tool_call["name"]):
            if random.random() < self.error_rate:
                return ToolMessage(
                    content="Error: Simulated tool failure",
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"],
                )
        return super().wrap_tool_call(request, handler)
```

### Deterministic Emulation

```python
class DeterministicEmulator(LLMToolEmulator):
    def __init__(self, *args, responses_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.responses = responses_map or {}

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call["name"]
        if tool_name in self.responses:
            return ToolMessage(
                content=self.responses[tool_name],
                tool_call_id=request.tool_call["id"],
                name=tool_name,
            )
        return super().wrap_tool_call(request, handler)
```

### Validation Emulation

```python
class ValidatingEmulator(LLMToolEmulator):
    def wrap_tool_call(self, request, handler):
        tool_args = request.tool_call["args"]

        # Add validation prompt
        prompt = (
            f"First, validate these arguments: {tool_args}\n"
            f"If invalid, return an error message.\n"
            f"If valid, generate a realistic tool response.\n"
            # ... rest of prompt
        )

        # Use enhanced prompt
        response = self.model.invoke([HumanMessage(prompt)])
        return ToolMessage(...)
```

### Logging and Metrics

```python
class InstrumentedEmulator(LLMToolEmulator):
    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call["name"]

        if should_emulate(tool_name):
            self.metrics.record_emulated_call(tool_name)
            result = super().wrap_tool_call(request, handler)
            self.metrics.record_emulated_result(tool_name, len(result.content))
            return result

        return handler(request)
```

### Hybrid Execution

```python
class HybridEmulator(LLMToolEmulator):
    """Emulate some calls, execute others based on runtime conditions."""

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call["name"]

        # Check runtime conditions
        if self.should_emulate_now(tool_name, request.context):
            return super().wrap_tool_call(request, handler)

        # Execute normally
        return handler(request)

    def should_emulate_now(self, tool_name, context):
        # Custom logic: time of day, resource availability, test mode, etc.
        if context.get("test_mode"):
            return True
        if is_business_hours() and tool_name in self.risky_tools:
            return True
        return False
```
