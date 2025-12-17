{{Infobox Implementation
| name = ConfigurableModel
| domain = LLM Operations
| sources = libs/langchain_v1/langchain/chat_models/base.py:L547-944
| last_updated = 2025-12-17
}}

== Overview ==

The <code>_ConfigurableModel</code> class is the implementation of the Model Declarative Operations principle. It provides a <code>Runnable</code> wrapper that enables runtime model configuration and deferred application of declarative operations, allowing users to specify model parameters and operations at execution time rather than initialization time.

== Description ==

<code>_ConfigurableModel</code> implements a sophisticated proxy system that:

1. '''Captures Configuration''': Stores default model parameters and configuration prefix
2. '''Queues Operations''': Records calls to declarative methods like <code>bind_tools()</code> and <code>with_structured_output()</code>
3. '''Merges Runtime Config''': Combines default configuration with runtime-provided configuration
4. '''Instantiates Models''': Creates concrete chat models using <code>_init_chat_model_helper</code> when needed
5. '''Applies Operations''': Executes queued operations on the instantiated model
6. '''Delegates Invocation''': Forwards execution requests to the configured model

The class maintains immutability by returning new instances for each configuration change, supporting functional composition patterns while managing stateful model instances internally.

== Code Reference ==

=== Class Definition and Initialization ===

<syntaxhighlight lang="python">
class _ConfigurableModel(Runnable[LanguageModelInput, Any]):
    def __init__(
        self,
        *,
        default_config: dict | None = None,
        configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
        config_prefix: str = "",
        queued_declarative_operations: Sequence[tuple[str, tuple, dict]] = (),
    ) -> None:
        self._default_config: dict = default_config or {}
        self._configurable_fields: Literal["any"] | list[str] = (
            configurable_fields if configurable_fields == "any" else list(configurable_fields)
        )
        self._config_prefix = (
            config_prefix + "_"
            if config_prefix and not config_prefix.endswith("_")
            else config_prefix
        )
        self._queued_declarative_operations: list[tuple[str, tuple, dict]] = list(
            queued_declarative_operations,
        )
</syntaxhighlight>

=== Operation Queuing via __getattr__ ===

<syntaxhighlight lang="python">
def __getattr__(self, name: str) -> Any:
    if name in _DECLARATIVE_METHODS:
        # Declarative operations that cannot be applied until after an actual model
        # object is instantiated. So instead of returning the actual operation,
        # we record the operation and its arguments in a queue. This queue is
        # then applied in order whenever we actually instantiate the model (in
        # self._model()).
        def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
            queued_declarative_operations = list(
                self._queued_declarative_operations,
            )
            queued_declarative_operations.append((name, args, kwargs))
            return _ConfigurableModel(
                default_config=dict(self._default_config),
                configurable_fields=list(self._configurable_fields)
                if isinstance(self._configurable_fields, list)
                else self._configurable_fields,
                config_prefix=self._config_prefix,
                queued_declarative_operations=queued_declarative_operations,
            )

        return queue
    if self._default_config and (model := self._model()) and hasattr(model, name):
        return getattr(model, name)
    msg = f"{name} is not a BaseChatModel attribute"
    if self._default_config:
        msg += " and is not implemented on the default model"
    msg += "."
    raise AttributeError(msg)
</syntaxhighlight>

=== Model Instantiation and Operation Application ===

<syntaxhighlight lang="python">
def _model(self, config: RunnableConfig | None = None) -> Runnable:
    params = {**self._default_config, **self._model_params(config)}
    model = _init_chat_model_helper(**params)
    for name, args, kwargs in self._queued_declarative_operations:
        model = getattr(model, name)(*args, **kwargs)
    return model

def _model_params(self, config: RunnableConfig | None) -> dict:
    config = ensure_config(config)
    model_params = {
        _remove_prefix(k, self._config_prefix): v
        for k, v in config.get("configurable", {}).items()
        if k.startswith(self._config_prefix)
    }
    if self._configurable_fields != "any":
        model_params = {k: v for k, v in model_params.items() if k in self._configurable_fields}
    return model_params
</syntaxhighlight>

=== Invocation Delegation ===

<syntaxhighlight lang="python">
@override
def invoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Any:
    return self._model(config).invoke(input, config=config, **kwargs)

@override
async def ainvoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    **kwargs: Any,
) -> Any:
    return await self._model(config).ainvoke(input, config=config, **kwargs)
</syntaxhighlight>

=== Declarative Method Stubs ===

<syntaxhighlight lang="python">
def bind_tools(
    self,
    tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
    return self.__getattr__("bind_tools")(tools, **kwargs)

def with_structured_output(
    self,
    schema: dict | type[BaseModel],
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict | BaseModel]:
    return self.__getattr__("with_structured_output")(schema, **kwargs)
</syntaxhighlight>

Source: <code>libs/langchain_v1/langchain/chat_models/base.py</code> lines 547-944

== I/O Contract ==

=== Initialization Parameters ===

; <code>default_config</code> : <code>dict | None</code>
: Default model configuration parameters. May include:
:* <code>model</code>: Model identifier
:* <code>model_provider</code>: Provider name
:* <code>temperature</code>: Temperature setting
:* Any other model-specific parameters

; <code>configurable_fields</code> : <code>Literal["any"] | list[str] | tuple[str, ...]</code>
: Controls which parameters can be configured at runtime:
:* <code>"any"</code>: All parameters are configurable (security warning applies)
:* <code>list[str]</code>: Only specified fields are configurable
:* Default: <code>"any"</code>

; <code>config_prefix</code> : <code>str</code>
: Prefix for configuration keys to avoid collisions. If specified, runtime configuration keys will be <code>{prefix}_{field}</code>

; <code>queued_declarative_operations</code> : <code>Sequence[tuple[str, tuple, dict]]</code>
: List of operations to apply, each as (method_name, args, kwargs)

=== Runtime Configuration ===

Runtime configuration is provided via the <code>config</code> parameter to invocation methods:

<syntaxhighlight lang="python">
config = {
    "configurable": {
        "model": "gpt-4o",
        "temperature": 0.7,
        # If config_prefix="foo", use:
        # "foo_model": "gpt-4o",
        # "foo_temperature": 0.7,
    }
}
</syntaxhighlight>

=== Declarative Methods ===

; <code>bind_tools(tools, **kwargs)</code>
: Queues tool binding operation. When model is instantiated, calls <code>model.bind_tools(tools, **kwargs)</code>

; <code>with_structured_output(schema, **kwargs)</code>
: Queues structured output operation. When model is instantiated, calls <code>model.with_structured_output(schema, **kwargs)</code>

=== Invocation Methods ===

All standard <code>Runnable</code> methods are supported:
* <code>invoke(input, config, **kwargs)</code> - Synchronous invocation
* <code>ainvoke(input, config, **kwargs)</code> - Asynchronous invocation
* <code>stream(input, config, **kwargs)</code> - Synchronous streaming
* <code>astream(input, config, **kwargs)</code> - Asynchronous streaming
* <code>batch(inputs, config, **kwargs)</code> - Batch processing
* <code>abatch(inputs, config, **kwargs)</code> - Asynchronous batch processing

== Usage Examples ==

=== Example 1: Basic Configurable Model ===

<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Create model without specifying which provider
configurable_model = init_chat_model(temperature=0)

# Use OpenAI at runtime
response1 = configurable_model.invoke(
    "What is 2+2?",
    config={"configurable": {"model": "gpt-4o"}}
)

# Use Anthropic at runtime
response2 = configurable_model.invoke(
    "What is 2+2?",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}}
)
</syntaxhighlight>

=== Example 2: Configurable Model with Default ===

<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Specify default model but allow runtime override
configurable_model = init_chat_model(
    "openai:gpt-4o",
    configurable_fields=("model", "model_provider", "temperature"),
    temperature=0
)

# Use default (OpenAI GPT-4o at temperature 0)
response1 = configurable_model.invoke("Hello!")

# Override to use Anthropic with higher temperature
response2 = configurable_model.invoke(
    "Hello!",
    config={
        "configurable": {
            "model": "claude-sonnet-4-5-20250929",
            "model_provider": "anthropic",
            "temperature": 0.8
        }
    }
)
</syntaxhighlight>

=== Example 3: Model with Config Prefix ===

<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Use prefix to distinguish multiple configurable models
summarizer = init_chat_model(
    config_prefix="summarizer",
    configurable_fields=("model", "model_provider"),
    temperature=0
)

analyzer = init_chat_model(
    config_prefix="analyzer",
    configurable_fields=("model", "model_provider"),
    temperature=0.5
)

# Configure both independently
config = {
    "configurable": {
        "summarizer_model": "gpt-4o",
        "summarizer_model_provider": "openai",
        "analyzer_model": "claude-sonnet-4-5-20250929",
        "analyzer_model_provider": "anthropic",
    }
}

summary = summarizer.invoke("Long text...", config=config)
analysis = analyzer.invoke(summary, config=config)
</syntaxhighlight>

=== Example 4: Binding Tools to Configurable Model ===

<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

# Create configurable model and bind tools
configurable_model = init_chat_model(
    "gpt-4o",
    configurable_fields=("model", "model_provider"),
    temperature=0
)

model_with_tools = configurable_model.bind_tools([GetWeather])

# Use with default model (GPT-4o)
response1 = model_with_tools.invoke("What's the weather in SF?")

# Use with different model at runtime
response2 = model_with_tools.invoke(
    "What's the weather in SF?",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}}
)
# Tools are bound to whichever model is selected at runtime
</syntaxhighlight>

=== Example 5: Structured Output with Configurable Model ===

<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Information about a person"""
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

# Create configurable model with structured output
configurable_model = init_chat_model(
    configurable_fields=("model", "model_provider")
)

structured_model = configurable_model.with_structured_output(Person)

# Use with OpenAI
person1 = structured_model.invoke(
    "Alice is 30 years old",
    config={"configurable": {"model": "gpt-4o"}}
)
# Returns: Person(name="Alice", age=30)

# Use with Anthropic
person2 = structured_model.invoke(
    "Bob is 25 years old",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}}
)
# Returns: Person(name="Bob", age=25)
</syntaxhighlight>

=== Example 6: Chaining Multiple Declarative Operations ===

<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """A search result"""
    query: str
    results: list[str]

class SearchTool(BaseModel):
    """Search the web"""
    query: str = Field(description="Search query")

# Chain multiple declarative operations
configurable_model = init_chat_model(
    configurable_fields=("model", "model_provider"),
    temperature=0
)

# First bind tools, then add structured output
model = (configurable_model
         .bind_tools([SearchTool])
         .with_structured_output(SearchResult))

# All operations applied to whatever model is selected at runtime
result = model.invoke(
    "Search for Python tutorials",
    config={"configurable": {"model": "gpt-4o"}}
)
</syntaxhighlight>

=== Example 7: Limited Configurability for Security ===

<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Only allow model selection, not API keys or endpoints
safe_configurable_model = init_chat_model(
    "openai:gpt-4o",
    configurable_fields=("model",),  # Only model is configurable
    api_key="your-secure-api-key",
    temperature=0
)

# User can change model
response = safe_configurable_model.invoke(
    "Hello!",
    config={"configurable": {"model": "gpt-3.5-turbo"}}
)
# Works: model is in configurable_fields

# But cannot change api_key
response = safe_configurable_model.invoke(
    "Hello!",
    config={"configurable": {"api_key": "malicious-key"}}
)
# api_key override is ignored because it's not in configurable_fields
</syntaxhighlight>

=== Example 8: Accessing Default Model Attributes ===

<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Model with default configuration
model_with_default = init_chat_model(
    "gpt-4o",
    configurable_fields=("temperature",),
    temperature=0
)

# Can access attributes of the default model
print(model_with_default.model_name)  # "gpt-4o"
print(model_with_default.temperature)  # 0

# Model without default has no attributes
model_no_default = init_chat_model(
    configurable_fields=("model", "model_provider")
)

try:
    print(model_no_default.model_name)
except AttributeError as e:
    print(e)  # "model_name is not a BaseChatModel attribute."
</syntaxhighlight>

== Implementation Details ==

=== Operation Queue Structure ===

Each queued operation is stored as a tuple:
<syntaxhighlight lang="python">
(method_name: str, args: tuple, kwargs: dict)
</syntaxhighlight>

Example queue:
<syntaxhighlight lang="python">
[
    ("bind_tools", ([WeatherTool, SearchTool],), {"tool_choice": "auto"}),
    ("with_structured_output", (SearchResult,), {"method": "json_mode"}),
]
</syntaxhighlight>

When the model is instantiated, operations are applied sequentially:
<syntaxhighlight lang="python">
model = _init_chat_model_helper(**params)
# Apply: model = model.bind_tools([WeatherTool, SearchTool], tool_choice="auto")
# Apply: model = model.with_structured_output(SearchResult, method="json_mode")
</syntaxhighlight>

=== Configuration Merging ===

Configuration is merged in priority order:
1. Default config (from initialization)
2. Runtime config (from <code>config["configurable"]</code>)

The <code>_model_params</code> method extracts runtime config:
<syntaxhighlight lang="python">
def _model_params(self, config: RunnableConfig | None) -> dict:
    config = ensure_config(config)
    # Extract config keys matching the prefix
    model_params = {
        _remove_prefix(k, self._config_prefix): v
        for k, v in config.get("configurable", {}).items()
        if k.startswith(self._config_prefix)
    }
    # Filter based on configurable_fields
    if self._configurable_fields != "any":
        model_params = {k: v for k, v in model_params.items() if k in self._configurable_fields}
    return model_params
</syntaxhighlight>

=== Prefix Handling ===

The constructor ensures prefixes end with underscore:
<syntaxhighlight lang="python">
self._config_prefix = (
    config_prefix + "_"
    if config_prefix and not config_prefix.endswith("_")
    else config_prefix
)
</syntaxhighlight>

This allows:
* <code>config_prefix=""</code> → no prefix, keys are <code>"model"</code>, <code>"temperature"</code>
* <code>config_prefix="foo"</code> → keys are <code>"foo_model"</code>, <code>"foo_temperature"</code>
* <code>config_prefix="foo_"</code> → keys are <code>"foo_model"</code>, <code>"foo_temperature"</code> (not <code>"foo__model"</code>)

=== Batch Processing Optimization ===

The <code>batch</code> and <code>abatch</code> methods include optimization for single-config batches:

<syntaxhighlight lang="python">
def batch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
    # If <= 1 config use the underlying models batch implementation.
    if config is None or isinstance(config, dict) or len(config) <= 1:
        if isinstance(config, list):
            config = config[0]
        return self._model(config).batch(
            inputs, config=config, return_exceptions=return_exceptions, **kwargs
        )
    # If multiple configs default to Runnable.batch which uses executor to invoke
    # in parallel.
    return super().batch(inputs, config=config, return_exceptions=return_exceptions, **kwargs)
</syntaxhighlight>

This ensures efficient batching when all inputs use the same model configuration, while supporting per-input model selection when needed.

=== Type Safety ===

The <code>InputType</code> property provides precise type information:

<syntaxhighlight lang="python">
@property
@override
def InputType(self) -> TypeAlias:
    from langchain_core.prompt_values import ChatPromptValueConcrete, StringPromptValue

    return str | StringPromptValue | ChatPromptValueConcrete | list[AnyMessage]
</syntaxhighlight>

This enables type checkers and IDEs to validate input types correctly.

== Related Pages ==

=== Principles ===
* [[langchain-ai_langchain_Model_Declarative_Operations|Model Declarative Operations]] - Principle implemented by this class

=== Related Implementations ===
* [[langchain-ai_langchain_init_chat_model_helper|init_chat_model_helper]] - Creates concrete models that operations are applied to
* [[langchain-ai_langchain_parse_model|parse_model]] - Parses model specifications from configuration

=== Workflows ===
* [[langchain-ai_langchain_Chat_Model_Initialization|Chat Model Initialization]] - Overall workflow utilizing configurable models

=== External References ===
* <code>Runnable</code> from langchain-core - Base interface implemented by this class
* <code>BaseChatModel</code> from langchain-core - Interface of instantiated models

[[Category:Implementations]]
[[Category:LLM Operations]]
[[Category:Design Patterns]]
[[Category:LangChain]]
