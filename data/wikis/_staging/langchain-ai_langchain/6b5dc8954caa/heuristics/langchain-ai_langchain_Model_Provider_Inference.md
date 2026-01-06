# Model Provider Inference Heuristic

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain Chat Models|https://github.com/langchain-ai/langchain/blob/main/libs/langchain_v1/langchain/chat_models/base.py]]
|-
! Domains
| [[domain::LLMs]], [[domain::Configuration]], [[domain::Developer_Experience]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Rules for automatic model provider detection from model names, enabling simplified `init_chat_model()` calls without explicit provider specification.

=== Description ===
The `init_chat_model()` function can automatically infer the model provider from model name prefixes. This heuristic documents the prefix-to-provider mapping and when explicit provider specification is required. Understanding these mappings reduces configuration errors and simplifies code.

=== Usage ===
Apply this heuristic when:
- Calling `init_chat_model()` without specifying `model_provider`
- Debugging "Unable to infer model provider" errors
- Switching between providers while maintaining code simplicity
- Using the `provider:model` shorthand syntax

== The Insight (Rule of Thumb) ==

* **Action:** Use model name prefixes to auto-detect provider, or use `provider:model` syntax
* **Values - Prefix Mappings:**
  - `gpt-*`, `o1*`, `o3*` → `openai`
  - `claude*` → `anthropic`
  - `command*` → `cohere`
  - `accounts/fireworks*` → `fireworks`
  - `gemini*` → `google_vertexai`
  - `amazon.*` → `bedrock`
  - `mistral*` → `mistralai`
  - `deepseek*` → `deepseek`
  - `grok*` → `xai`
  - `sonar*` → `perplexity`
  - `solar*` → `upstage`
* **Trade-off:** Auto-inference is convenient but may fail for new/custom models; explicit `model_provider` is always reliable
* **Alternative:** Use `provider:model` syntax (e.g., `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-5-20250929"`)

=== Supported Providers ===

| Provider Key | Package | Example Models |
|--------------|---------|----------------|
| `openai` | langchain-openai | gpt-4o, o3-mini |
| `anthropic` | langchain-anthropic | claude-sonnet-4-5-20250929 |
| `azure_openai` | langchain-openai | (requires explicit provider) |
| `google_vertexai` | langchain-google-vertexai | gemini-2.5-flash |
| `google_genai` | langchain-google-genai | (requires explicit provider) |
| `bedrock` | langchain-aws | amazon.titan |
| `cohere` | langchain-cohere | command-r |
| `mistralai` | langchain-mistralai | mistral-large |
| `ollama` | langchain-ollama | (requires explicit provider) |
| `deepseek` | langchain-deepseek | deepseek-chat |
| `xai` | langchain-xai | grok-2 |
| `perplexity` | langchain-perplexity | sonar-pro |
| `upstage` | langchain-upstage | solar-pro |

== Reasoning ==

Auto-inference works by checking model name prefixes against known patterns. This was designed to match the most common naming conventions used by each provider. However, not all providers have unique prefixes (e.g., Ollama models can be any name), so explicit specification is sometimes required.

Code Evidence from `chat_models/base.py:489-512`:
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

Provider:model parsing from `chat_models/base.py:515-530`:
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

== Related Pages ==
* [[uses_heuristic::Implementation:langchain-ai_langchain_model_parsing_functions]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_init_chat_model]]
* [[uses_heuristic::Implementation:langchain-ai_langchain_init_chat_model_helper]]
* [[uses_heuristic::Workflow:langchain-ai_langchain_Chat_Model_Initialization_Workflow]]
* [[uses_heuristic::Principle:langchain-ai_langchain_Model_Identifier_Parsing]]
