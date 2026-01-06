{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Tools|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Doc|LangChain Core Tools|https://api.python.langchain.com/en/latest/tools/langchain_core.tools.base.BaseTool.html]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Tool_Calling]], [[domain::Agents]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Wrapper documentation for LangChain Core's tool abstractions (`BaseTool`, `StructuredTool`) used for defining agent capabilities in LangChain workflows.

=== Description ===

`BaseTool` and `StructuredTool` are the foundational classes from `langchain_core` for defining tools that agents can call. While these classes are defined in the external `langchain_core` package, they are the primary interface for tool definition in LangChain agent workflows.

* **BaseTool:** Abstract base class for all tools with `name`, `description`, and `args_schema`
* **StructuredTool:** Concrete implementation that accepts a function, Pydantic model, or JSON schema to define tool parameters

In agent creation workflows, tools are converted to this format before being bound to models for tool calling.

=== Usage ===

Use these classes when:
* Defining custom tools for agents to call
* Wrapping Python functions as agent tools
* Creating tools with typed input schemas (Pydantic models)
* Building tool-using agent workflows

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/langchain-ai/langchain langchain-core]
* '''LangChain Entry Point:''' libs/langchain_v1/langchain/tools/__init__.py
* '''Core Implementation:''' langchain_core.tools.BaseTool, langchain_core.tools.StructuredTool

=== Signature ===
<syntaxhighlight lang="python">
class BaseTool(ABC):
    """Abstract base class for tools."""
    name: str
    """The unique name of the tool for identification."""

    description: str
    """Human-readable description of what the tool does."""

    args_schema: Type[BaseModel] | dict
    """Schema for tool arguments (Pydantic model or JSON schema)."""

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """Execute the tool (sync)."""

    async def _arun(self, *args, **kwargs) -> Any:
        """Execute the tool (async)."""


class StructuredTool(BaseTool):
    """Tool that accepts structured inputs via schema."""

    def __init__(
        self,
        args_schema: dict | Type[BaseModel],
        name: str,
        description: str,
        func: Callable | None = None,
        coroutine: Callable | None = None,
    ) -> None:
        """Create a StructuredTool.

        Args:
            args_schema: JSON schema dict or Pydantic model for inputs.
            name: Tool name for model binding.
            description: Tool description for model context.
            func: Sync function to execute.
            coroutine: Async function to execute.
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.tools import BaseTool, StructuredTool
# Or directly from core:
from langchain_core.tools import BaseTool, StructuredTool
</syntaxhighlight>

== I/O Contract ==

=== Inputs (StructuredTool Constructor) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| args_schema || dict | Type[BaseModel] || Yes || JSON schema or Pydantic model defining tool inputs
|-
| name || str || Yes || Unique tool name for model binding
|-
| description || str || Yes || Description explaining tool purpose for model
|-
| func || Callable | None || No || Sync function to execute
|-
| coroutine || Callable | None || No || Async function to execute
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tool instance || BaseTool || Configured tool ready for agent binding
|}

== Usage Examples ==

=== Using Pydantic Model Schema ===
<syntaxhighlight lang="python">
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    """Input for weather lookup."""
    location: str = Field(description="City and state, e.g., 'San Francisco, CA'")
    unit: str = Field(default="celsius", description="Temperature unit")


def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: 72°{unit[0].upper()}"


weather_tool = StructuredTool(
    args_schema=WeatherInput,
    name="get_weather",
    description="Get current weather for a location",
    func=get_weather,
)
</syntaxhighlight>

=== Using @tool Decorator ===
<syntaxhighlight lang="python">
from langchain.tools import tool
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(description="Search query string")


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


# search is now a StructuredTool instance
</syntaxhighlight>

=== Binding Tools to Agent ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# Define tools
tools = [weather_tool, search_tool]

# Create agent with tools
agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=tools,
)

# Invoke agent - it can now call the tools
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in NYC?"}]})
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Tool_Definition]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
