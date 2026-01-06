{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::NLP]], [[domain::LLM]], [[domain::Provider_Abstraction]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Concrete tool for parsing model identifier strings and inferring provider from model name prefixes, provided by LangChain's chat model factory.

=== Description ===

`_parse_model` and `_attempt_infer_model_provider` are internal functions that handle model identifier parsing:
* Parse `provider:model` syntax (e.g., "openai:gpt-4o")
* Infer provider from model name prefixes (e.g., "gpt-" → openai)
* Normalize provider names (convert hyphens to underscores, lowercase)

These functions enable the unified `init_chat_model` interface where users don't need to know provider-specific class names.

=== Usage ===

Use these functions (indirectly via `init_chat_model`) when:
* Initializing models without explicit provider specification
* Using the `provider:model` syntax
* Building dynamic model selection systems

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/chat_models/base.py
* '''Lines:''' L489-530

=== Signature ===
<syntaxhighlight lang="python">
def _attempt_infer_model_provider(model_name: str) -> str | None:
    """Attempt to infer model provider from model name prefix.

    Args:
        model_name: The model name to analyze

    Returns:
        Provider name if inferrable, None otherwise

    Inference rules:
        - "gpt-", "o1", "o3" → "openai"
        - "claude" → "anthropic"
        - "command" → "cohere"
        - "accounts/fireworks" → "fireworks"
        - "gemini" → "google_vertexai"
        - "amazon." → "bedrock"
        - "mistral" → "mistralai"
        - "deepseek" → "deepseek"
        - "grok" → "xai"
        - "sonar" → "perplexity"
        - "solar" → "upstage"
    """


def _parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    """Parse model string and resolve provider.

    Args:
        model: Model identifier (e.g., "gpt-4o" or "openai:gpt-4o")
        model_provider: Explicit provider override (optional)

    Returns:
        Tuple of (model_name, provider_name)

    Raises:
        ValueError: If provider cannot be determined
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# These are internal functions, use via init_chat_model
from langchain.chat_models import init_chat_model

# Direct usage (not recommended):
from langchain.chat_models.base import _parse_model, _attempt_infer_model_provider
</syntaxhighlight>

== I/O Contract ==

=== Inputs (_attempt_infer_model_provider) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || Yes || Model name to analyze for provider inference
|}

=== Outputs (_attempt_infer_model_provider) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || str | None || Inferred provider name or None if cannot infer
|}

=== Inputs (_parse_model) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || Model identifier string
|-
| model_provider || str | None || No || Explicit provider override
|}

=== Outputs (_parse_model) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || tuple[str, str] || (model_name, provider_name) tuple
|}

== Usage Examples ==

=== Provider Inference Examples ===
<syntaxhighlight lang="python">
from langchain.chat_models.base import _attempt_infer_model_provider

# OpenAI models
assert _attempt_infer_model_provider("gpt-4o") == "openai"
assert _attempt_infer_model_provider("gpt-3.5-turbo") == "openai"
assert _attempt_infer_model_provider("o1-preview") == "openai"
assert _attempt_infer_model_provider("o3-mini") == "openai"

# Anthropic models
assert _attempt_infer_model_provider("claude-3-opus") == "anthropic"
assert _attempt_infer_model_provider("claude-sonnet-4-5-20250929") == "anthropic"

# Google models
assert _attempt_infer_model_provider("gemini-pro") == "google_vertexai"
assert _attempt_infer_model_provider("gemini-2.5-flash") == "google_vertexai"

# Other providers
assert _attempt_infer_model_provider("command-r-plus") == "cohere"
assert _attempt_infer_model_provider("mistral-large") == "mistralai"
assert _attempt_infer_model_provider("deepseek-chat") == "deepseek"

# Unknown prefix returns None
assert _attempt_infer_model_provider("custom-model") is None
</syntaxhighlight>

=== Model Parsing Examples ===
<syntaxhighlight lang="python">
from langchain.chat_models.base import _parse_model

# Explicit provider:model syntax
model, provider = _parse_model("openai:gpt-4o", None)
assert model == "gpt-4o"
assert provider == "openai"

# Provider inference
model, provider = _parse_model("gpt-4o", None)
assert model == "gpt-4o"
assert provider == "openai"

# Explicit provider override
model, provider = _parse_model("my-custom-model", "openai")
assert model == "my-custom-model"
assert provider == "openai"

# Provider normalization
model, provider = _parse_model("model", "AZURE-OPENAI")
assert provider == "azure_openai"  # Normalized
</syntaxhighlight>

=== Using via init_chat_model ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# All these are equivalent:
model1 = init_chat_model("gpt-4o")  # Provider inferred
model2 = init_chat_model("openai:gpt-4o")  # Provider:model syntax
model3 = init_chat_model("gpt-4o", model_provider="openai")  # Explicit provider

# Works across providers:
anthropic = init_chat_model("claude-sonnet-4-5-20250929")  # Inferred as anthropic
google = init_chat_model("gemini-pro")  # Inferred as google_vertexai
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# Unknown model without provider raises ValueError
try:
    model = init_chat_model("unknown-custom-model")
except ValueError as e:
    print(e)  # "Unable to infer model provider for model='unknown-custom-model'"

# Solution: Specify provider explicitly
model = init_chat_model("unknown-custom-model", model_provider="openai")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Model_Identifier_Parsing]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
