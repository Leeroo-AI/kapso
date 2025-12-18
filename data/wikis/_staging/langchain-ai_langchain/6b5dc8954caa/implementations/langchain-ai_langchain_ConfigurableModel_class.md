{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Runtime_Configuration]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for creating runtime-configurable chat model wrappers that defer instantiation until invocation, provided by LangChain.

=== Description ===

`_ConfigurableModel` is a `Runnable` wrapper that enables runtime model switching. Instead of instantiating a specific model at creation time, it stores configuration and instantiates the actual model when `invoke()` is called, using parameters from `config["configurable"]`.

Key capabilities:
* Defer model selection to runtime
* Override model parameters per invocation
* Queue declarative operations (`bind_tools`, `with_structured_output`)
* Support A/B testing and dynamic model routing

=== Usage ===

Use `_ConfigurableModel` when:
* Building applications that switch models at runtime
* Implementing A/B testing for model selection
* Creating user-selectable model interfaces
* Developing model routing/fallback systems

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/chat_models/base.py
* '''Lines:''' L547-648

=== Signature ===
<syntaxhighlight lang="python">
class _ConfigurableModel(Runnable[LanguageModelInput, Any]):
    """A configurable chat model wrapper.

    Instantiates the actual model at runtime based on config.
    """

    def __init__(
        self,
        *,
        default_config: dict | None = None,
        configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
        config_prefix: str = "",
        queued_declarative_operations: Sequence[tuple[str, tuple, dict]] = (),
    ) -> None:
        """Create a configurable model.

        Args:
            default_config: Default model configuration (model, provider, kwargs)
            configurable_fields: Which fields can be overridden at runtime
                - "any": All fields configurable
                - list/tuple: Only specified fields configurable
            config_prefix: Prefix for config keys (e.g., "llm" -> "llm_model")
            queued_declarative_operations: Operations to apply after instantiation
        """

    def invoke(self, input, config=None, **kwargs):
        """Invoke with runtime configuration."""

    async def ainvoke(self, input, config=None, **kwargs):
        """Async invoke with runtime configuration."""

    def stream(self, input, config=None, **kwargs):
        """Stream with runtime configuration."""

    async def astream(self, input, config=None, **kwargs):
        """Async stream with runtime configuration."""

    def bind_tools(self, tools, **kwargs) -> "_ConfigurableModel":
        """Queue tool binding for later application."""

    def with_structured_output(self, schema, **kwargs) -> "_ConfigurableModel":
        """Queue structured output for later application."""

    def with_config(self, config=None, **kwargs) -> "_ConfigurableModel":
        """Create new configurable with merged config."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Create configurable model
configurable = init_chat_model(configurable_fields=("model", "model_provider"))
# Returns _ConfigurableModel instance
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Constructor) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| default_config || dict | None || No || Default parameters for model instantiation
|-
| configurable_fields || "any" | list[str] || No || Which fields can be overridden at runtime
|-
| config_prefix || str || No || Prefix for config keys (default: "")
|-
| queued_declarative_operations || Sequence[tuple] || No || Operations to apply after instantiation
|}

=== Inputs (invoke) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input || LanguageModelInput || Yes || Messages to send to model
|-
| config || RunnableConfig | None || No || Runtime config with "configurable" dict
|}

=== Config Keys ===
{| class="wikitable"
|-
! Key (with prefix "foo") !! Type !! Description
|-
| foo_model || str || Model name to use
|-
| foo_model_provider || str || Provider to use
|-
| foo_temperature || float || Temperature override
|-
| foo_max_tokens || int || Max tokens override
|}

== Usage Examples ==

=== Basic Configurable Model ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Create configurable model (no default)
model = init_chat_model(configurable_fields=("model", "model_provider"))

# Use different models at runtime
response_gpt = model.invoke(
    "Hello!",
    config={"configurable": {"model": "gpt-4o"}}
)

response_claude = model.invoke(
    "Hello!",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}}
)
</syntaxhighlight>

=== With Default and Override ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Default model with configurable override
model = init_chat_model(
    "gpt-4o",
    configurable_fields="any",
    temperature=0,
)

# Use default
response = model.invoke("Hello!")  # Uses gpt-4o, temp=0

# Override at runtime
response = model.invoke(
    "Hello!",
    config={"configurable": {
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }}
)
</syntaxhighlight>

=== With Config Prefix ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Prefix for namespacing
model = init_chat_model(
    "gpt-4o",
    configurable_fields="any",
    config_prefix="primary",
)

# Config keys are prefixed
response = model.invoke(
    "Hello!",
    config={"configurable": {
        "primary_model": "gpt-4o-mini",  # Note: primary_ prefix
        "primary_temperature": 0.5
    }}
)
</syntaxhighlight>

=== With Tool Binding ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
from pydantic import BaseModel


class Weather(BaseModel):
    """Get weather for a location."""
    location: str


# Tools bound to configurable model
model = init_chat_model(configurable_fields=("model",))
model_with_tools = model.bind_tools([Weather])

# Tools work with any configured model
response = model_with_tools.invoke(
    "What's the weather in NYC?",
    config={"configurable": {"model": "gpt-4o"}}
)
</syntaxhighlight>

=== Restricted Configurable Fields ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Only model and temperature can be configured
# api_key, base_url, etc. are NOT configurable (security)
model = init_chat_model(
    "gpt-4o",
    configurable_fields=["model", "temperature"],  # Explicit list
    api_key="sk-...",  # Fixed, not configurable
)

# This works:
model.invoke("Hi", config={"configurable": {"temperature": 0.5}})

# This is ignored (api_key not in configurable_fields):
model.invoke("Hi", config={"configurable": {"api_key": "different-key"}})
</syntaxhighlight>

=== A/B Testing Pattern ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
import random

model = init_chat_model(configurable_fields=("model",))


def get_config_for_user(user_id: str) -> dict:
    """A/B test: 50% GPT-4, 50% Claude."""
    if hash(user_id) % 2 == 0:
        return {"configurable": {"model": "gpt-4o"}}
    else:
        return {"configurable": {"model": "claude-sonnet-4-5-20250929"}}


# Each user gets consistent model assignment
response = model.invoke("Hello!", config=get_config_for_user(user_id))
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Configurable_Model_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
