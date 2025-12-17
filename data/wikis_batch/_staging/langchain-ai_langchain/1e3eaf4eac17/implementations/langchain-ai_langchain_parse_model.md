{{Infobox Implementation
| name = parse_model
| domain = LLM Operations
| sources = libs/langchain_v1/langchain/chat_models/base.py:L515-530
| last_updated = 2025-12-17
}}

== Overview ==

The <code>_parse_model</code> function is the implementation of the Model String Parsing principle, responsible for extracting and normalizing the model name and provider from flexible input formats. It handles prefixed model specifications, provider inference, and normalization to ensure consistent downstream processing.

== Description ==

<code>_parse_model</code> processes a model string and optional provider parameter to produce a normalized (model_name, provider) tuple. It implements a three-tier resolution strategy:

1. '''Prefix Parsing''': If the model string contains ":" and the prefix matches a supported provider, extract the provider and remaining model name
2. '''Explicit Provider''': If <code>model_provider</code> is specified, use it directly
3. '''Inference''': Call <code>_attempt_infer_model_provider()</code> to detect provider from model name patterns

After provider resolution, the function normalizes the provider name by converting hyphens to underscores and converting to lowercase, ensuring consistency with internal provider naming conventions.

== Code Reference ==

<syntaxhighlight lang="python">
def _parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    if (
        not model_provider
        and ":" in model
        and model.split(":", maxsplit=1)[0] in _SUPPORTED_PROVIDERS
    ):
        model_provider = model.split(":", maxsplit=1)[0]
        model = ":".join(model.split(":")[1:])
    model_provider = model_provider or _attempt_infer_model_provider(model)
    if not model_provider:
        msg = (
            f"Unable to infer model provider for {model=}, please specify model_provider directly."
        )
        raise ValueError(msg)
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider
</syntaxhighlight>

Source: <code>libs/langchain_v1/langchain/chat_models/base.py</code> lines 515-530

=== Supporting Function: _attempt_infer_model_provider ===

<syntaxhighlight lang="python">
def _attempt_infer_model_provider(model_name: str) -> str | None:
    if any(model_name.startswith(pre) for pre in ("gpt-", "o1", "o3")):
        return "openai"
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("command"):
        return "cohere"
    if model_name.startswith("accounts/fireworks"):
        return "fireworks"
    if model_name.startswith("gemini"):
        return "google_vertexai"
    if model_name.startswith("amazon."):
        return "bedrock"
    if model_name.startswith("mistral"):
        return "mistralai"
    if model_name.startswith("deepseek"):
        return "deepseek"
    if model_name.startswith("grok"):
        return "xai"
    if model_name.startswith("sonar"):
        return "perplexity"
    if model_name.startswith("solar"):
        return "upstage"
    return None
</syntaxhighlight>

Source: <code>libs/langchain_v1/langchain/chat_models/base.py</code> lines 489-512

== I/O Contract ==

=== Input Parameters ===

; <code>model</code> : <code>str</code>
: The model identifier, which may include a provider prefix (e.g., "openai:gpt-4o") or be a plain model name (e.g., "gpt-4o")

; <code>model_provider</code> : <code>str | None</code>
: Optional explicit provider specification. If provided, takes precedence over prefix parsing and inference

=== Return Value ===

; <code>tuple[str, str]</code>
: A tuple of (model_name, normalized_provider) where:
:* <code>model_name</code> is the model identifier with provider prefix removed
:* <code>normalized_provider</code> is the provider name in lowercase with underscores

=== Exceptions ===

; <code>ValueError</code>
: Raised when provider cannot be determined from any source (prefix, explicit parameter, or inference)

=== Supported Providers ===

The function recognizes the following providers for prefix parsing:
* openai
* anthropic
* azure_openai
* azure_ai
* cohere
* google_vertexai
* google_genai
* fireworks
* ollama
* together
* mistralai
* huggingface
* groq
* bedrock
* bedrock_converse
* google_anthropic_vertex
* deepseek
* ibm
* nvidia
* xai
* perplexity
* upstage

=== Inference Patterns ===

The following model name prefixes trigger automatic provider inference:
* "gpt-", "o1", "o3" → openai
* "claude" → anthropic
* "command" → cohere
* "accounts/fireworks" → fireworks
* "gemini" → google_vertexai
* "amazon." → bedrock
* "mistral" → mistralai
* "deepseek" → deepseek
* "grok" → xai
* "sonar" → perplexity
* "solar" → upstage

== Usage Examples ==

=== Example 1: Prefixed Model Specification ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _parse_model

# Parse model with explicit provider prefix
model, provider = _parse_model("openai:gpt-4o", model_provider=None)
# Result: ("gpt-4o", "openai")

# Parse anthropic model with prefix
model, provider = _parse_model("anthropic:claude-sonnet-4-5-20250929", None)
# Result: ("claude-sonnet-4-5-20250929", "anthropic")
</syntaxhighlight>

=== Example 2: Provider Inference ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _parse_model

# Infer OpenAI from model name
model, provider = _parse_model("gpt-4o", model_provider=None)
# Result: ("gpt-4o", "openai")

# Infer Anthropic from model name
model, provider = _parse_model("claude-3-opus", None)
# Result: ("claude-3-opus", "anthropic")

# Infer from newer model names
model, provider = _parse_model("o1-preview", None)
# Result: ("o1-preview", "openai")
</syntaxhighlight>

=== Example 3: Explicit Provider Parameter ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _parse_model

# Explicit provider overrides inference
model, provider = _parse_model("my-custom-model", model_provider="ollama")
# Result: ("my-custom-model", "ollama")

# Explicit provider with prefixed model (provider param wins)
model, provider = _parse_model("openai:gpt-4o", model_provider="azure_openai")
# Result: ("openai:gpt-4o", "azure_openai")
# Note: prefix is not removed when explicit provider is given
</syntaxhighlight>

=== Example 4: Provider Normalization ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _parse_model

# Hyphens converted to underscores
model, provider = _parse_model("gpt-4o", model_provider="azure-openai")
# Result: ("gpt-4o", "azure_openai")

# Case normalization
model, provider = _parse_model("model", model_provider="OpenAI")
# Result: ("model", "openai")
</syntaxhighlight>

=== Example 5: Error Cases ===

<syntaxhighlight lang="python">
from langchain.chat_models.base import _parse_model

# Provider cannot be inferred
try:
    model, provider = _parse_model("unknown-model", model_provider=None)
except ValueError as e:
    print(e)
    # "Unable to infer model provider for model='unknown-model',
    #  please specify model_provider directly."

# Unsupported prefix (not in _SUPPORTED_PROVIDERS)
model, provider = _parse_model("unsupported:model-name", None)
# Result: Attempts inference on full string "unsupported:model-name"
# Likely raises ValueError unless it matches an inference pattern
</syntaxhighlight>

== Related Pages ==

=== Principles ===
* [[langchain-ai_langchain_Model_String_Parsing|Model String Parsing]] - Principle implemented by this function

=== Related Implementations ===
* [[langchain-ai_langchain_check_pkg|check_pkg]] - Validates parsed provider has required packages
* [[langchain-ai_langchain_init_chat_model_helper|init_chat_model_helper]] - Uses parsed results for model instantiation

=== Workflows ===
* [[langchain-ai_langchain_Chat_Model_Initialization|Chat Model Initialization]] - Overall workflow utilizing this parsing

[[Category:Implementations]]
[[Category:LLM Operations]]
[[Category:String Parsing]]
[[Category:LangChain]]
