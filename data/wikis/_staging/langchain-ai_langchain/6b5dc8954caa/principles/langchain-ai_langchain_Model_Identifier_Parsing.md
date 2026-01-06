{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
|-
! Domains
| [[domain::LLM]], [[domain::Provider_Abstraction]], [[domain::API_Design]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Pattern for extracting model name and provider information from unified identifier strings.

=== Description ===

Model Identifier Parsing is the process of interpreting model specification strings to determine both the model name and the provider to use. This enables a developer-friendly API where:
* Users don't need to import provider-specific classes
* Model switching requires only changing a string
* Provider can be inferred from well-known model name patterns

The pattern supports multiple specification formats:
* Explicit: `"provider:model"` (e.g., `"openai:gpt-4o"`)
* Inferred: `"model"` with prefix matching (e.g., `"gpt-4o"` → OpenAI)
* Override: `"model"` + explicit `model_provider` parameter

=== Usage ===

Use Model Identifier Parsing when:
* Building model-agnostic applications
* Creating configuration-driven model selection
* Implementing model routing/switching logic
* Designing user-facing model selection interfaces

== Theoretical Basis ==

Model Identifier Parsing implements **Convention over Configuration** and **Provider Abstraction** patterns.

'''1. Identifier Format Grammar'''

<syntaxhighlight lang="text">
model_identifier ::= provider_syntax | model_name

provider_syntax ::= provider_name ":" model_name

provider_name ::= [a-z_]+   # e.g., "openai", "azure_openai"
model_name ::= [a-zA-Z0-9_.-]+  # e.g., "gpt-4o", "claude-sonnet-4-5-20250929"

Examples:
  "gpt-4o"                    → model_name (provider inferred)
  "openai:gpt-4o"             → provider_syntax (explicit)
  "azure_openai:gpt-4"        → provider_syntax (explicit)
</syntaxhighlight>

'''2. Provider Inference Rules'''

<syntaxhighlight lang="python">
# Pseudo-code for inference logic
PROVIDER_PREFIXES = [
    # (prefix_pattern, provider)
    (r"^gpt-", "openai"),
    (r"^o[13]", "openai"),  # o1, o3 models
    (r"^claude", "anthropic"),
    (r"^gemini", "google_vertexai"),
    (r"^command", "cohere"),
    (r"^accounts/fireworks", "fireworks"),
    (r"^amazon\.", "bedrock"),
    (r"^mistral", "mistralai"),
    (r"^deepseek", "deepseek"),
    (r"^grok", "xai"),
    (r"^sonar", "perplexity"),
    (r"^solar", "upstage"),
]

def infer_provider(model_name: str) -> str | None:
    for prefix, provider in PROVIDER_PREFIXES:
        if re.match(prefix, model_name):
            return provider
    return None
</syntaxhighlight>

'''3. Resolution Priority'''

<syntaxhighlight lang="python">
# Resolution order:
# 1. Explicit model_provider parameter (highest priority)
# 2. Provider in "provider:model" syntax
# 3. Inferred from model name prefix
# 4. Error if none work

def resolve_provider(model: str, model_provider: str | None) -> tuple[str, str]:
    # Priority 1: Explicit parameter
    if model_provider:
        return model, normalize(model_provider)

    # Priority 2: provider:model syntax
    if ":" in model:
        parts = model.split(":", 1)
        if parts[0] in SUPPORTED_PROVIDERS:
            return parts[1], normalize(parts[0])

    # Priority 3: Inference
    inferred = infer_provider(model)
    if inferred:
        return model, inferred

    # Priority 4: Error
    raise ValueError(f"Cannot determine provider for {model}")
</syntaxhighlight>

'''4. Normalization'''

<syntaxhighlight lang="python">
def normalize(provider: str) -> str:
    """Normalize provider string."""
    # Convert to lowercase
    provider = provider.lower()
    # Replace hyphens with underscores
    provider = provider.replace("-", "_")
    return provider

# Examples:
# "OpenAI" → "openai"
# "azure-openai" → "azure_openai"
# "GOOGLE_VERTEXAI" → "google_vertexai"
</syntaxhighlight>

'''5. Error Messages'''

<syntaxhighlight lang="python">
# Good error messages are critical for developer experience
def parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    model_name, provider = _try_parse(model, model_provider)

    if not provider:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Unable to infer model provider for model={model!r}, "
            f"please specify model_provider directly.\n\n"
            f"Supported providers: {supported}"
        )

    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Unsupported model_provider={provider!r}.\n\n"
            f"Supported providers: {supported}"
        )

    return model_name, provider
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_model_parsing_functions]]

=== Used By Workflows ===
* Chat_Model_Initialization_Workflow (Step 1)
