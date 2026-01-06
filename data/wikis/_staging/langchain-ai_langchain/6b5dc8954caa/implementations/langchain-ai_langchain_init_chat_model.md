{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Docs|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Chat_Models]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for initializing chat models from any supported provider using a unified factory interface, provided by LangChain.

=== Description ===

`init_chat_model` is LangChain's universal chat model factory function that abstracts away provider-specific initialization logic. It supports 20+ providers including OpenAI, Anthropic, Google, AWS Bedrock, and more. The function can infer providers from model name prefixes (e.g., "gpt-" → OpenAI, "claude" → Anthropic) or accept explicit provider specification using the `provider:model` syntax.

Key capabilities:
* Provider inference from model name prefixes
* Runtime-configurable models for dynamic provider/model switching
* Declarative operation queuing (`bind_tools`, `with_structured_output`)
* Consistent interface across all providers

=== Usage ===

Use this function when:
* Initializing chat models in agent workflows
* Building applications that need to switch between providers dynamically
* Creating configurable LLM pipelines where the model can be changed at runtime

NOT for: Direct low-level API access to specific providers (use provider packages directly).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/chat_models/base.py
* '''Lines:''' L59-329

=== Signature ===
<syntaxhighlight lang="python">
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel | _ConfigurableModel:
    """Initialize a chat model from any supported provider using a unified interface.

    Args:
        model: The name or ID of the model, e.g. 'o3-mini', 'claude-sonnet-4-5-20250929'.
            Can use '{provider}:{model}' format.
        model_provider: Explicit provider override (openai, anthropic, etc.).
        configurable_fields: Which model parameters are configurable at runtime.
            None = fixed model, 'any' = all fields, list = specified fields.
        config_prefix: Namespace prefix for configurable keys.
        **kwargs: Provider-specific kwargs (temperature, max_tokens, etc.).

    Returns:
        BaseChatModel if fixed, _ConfigurableModel if configurable.

    Raises:
        ValueError: If model_provider cannot be inferred or isn't supported.
        ImportError: If provider integration package is not installed.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str | None || No || Model name/ID (e.g., "gpt-4o", "claude-sonnet-4-5-20250929", "openai:o3-mini")
|-
| model_provider || str | None || No || Explicit provider override (openai, anthropic, google_vertexai, etc.)
|-
| configurable_fields || Literal["any"] | list[str] | None || No || Fields configurable at runtime
|-
| config_prefix || str | None || No || Namespace prefix for config keys
|-
| **kwargs || Any || No || Provider-specific parameters (temperature, max_tokens, api_key)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || BaseChatModel | _ConfigurableModel || Ready-to-use chat model instance or configurable model wrapper
|}

== Usage Examples ==

=== Fixed Model Initialization ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Initialize with provider inference from model name
gpt4 = init_chat_model("gpt-4o", temperature=0)
claude = init_chat_model("claude-sonnet-4-5-20250929", temperature=0)

# Initialize with explicit provider:model syntax
model = init_chat_model("openai:o3-mini", temperature=0.7)

# Use the model
response = model.invoke("Hello, how are you?")
</syntaxhighlight>

=== Configurable Model for Runtime Switching ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Create a configurable model (no default model)
configurable_model = init_chat_model(temperature=0)

# Switch models at runtime via config
response1 = configurable_model.invoke(
    "What's the weather?",
    config={"configurable": {"model": "gpt-4o"}}
)

response2 = configurable_model.invoke(
    "What's the weather?",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}}
)
</syntaxhighlight>

=== With Default and Runtime Override ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Model with default that can be overridden
model = init_chat_model(
    "openai:gpt-4o",
    configurable_fields="any",
    config_prefix="llm",
    temperature=0,
)

# Use default
model.invoke("Hello")

# Override at runtime
model.invoke(
    "Hello",
    config={"configurable": {"llm_model": "anthropic:claude-sonnet-4-5-20250929", "llm_temperature": 0.5}}
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Chat_Model_Initialization]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
