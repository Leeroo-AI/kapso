# LangChain Runtime Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Installation|https://docs.langchain.com/oss/python/langchain/installation]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::LLMs]], [[domain::AI_Agents]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Python 3.9+ environment with `langchain-core` as the base framework and provider-specific integration packages for chat model access.

=== Description ===
This environment provides the core runtime context for LangChain applications. It includes the base abstractions (`langchain-core`), the main `langchain` package with agent factories and middleware, and optional provider integration packages for specific LLM providers (OpenAI, Anthropic, etc.). The framework supports both synchronous and asynchronous execution patterns.

=== Usage ===
Use this environment for any LangChain-based application including:
- **Chat Model Initialization** via `init_chat_model()`
- **Agent Creation** via `create_agent()`
- **Tool Definition** and middleware composition
- **Structured Output** workflows

This is the mandatory prerequisite for running all workflow implementations in this wiki.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows || Cross-platform Python
|-
| Python || 3.9+ || Required for type hints and async features
|-
| Disk || Minimal || Provider packages vary; ~50MB base
|}

== Dependencies ==

=== Core Packages ===
* `langchain-core` >= 0.3.0 - Base abstractions and interfaces
* `langchain` >= 0.3.0 - Main package with agent factory
* `pydantic` >= 2.0 - Data validation for schemas
* `langgraph` >= 0.2.0 - StateGraph for agent execution

=== Provider Packages (Install as needed) ===

The `init_chat_model()` function dynamically imports provider packages. Install only the ones you need:

| Provider | Package | Install Command |
|----------|---------|-----------------|
| OpenAI | `langchain-openai` | `pip install langchain-openai` |
| Anthropic | `langchain-anthropic` | `pip install langchain-anthropic` |
| Azure OpenAI | `langchain-openai` | `pip install langchain-openai` |
| Google Vertex AI | `langchain-google-vertexai` | `pip install langchain-google-vertexai` |
| Google GenAI | `langchain-google-genai` | `pip install langchain-google-genai` |
| AWS Bedrock | `langchain-aws` | `pip install langchain-aws` |
| Cohere | `langchain-cohere` | `pip install langchain-cohere` |
| Fireworks | `langchain-fireworks` | `pip install langchain-fireworks` |
| Together | `langchain-together` | `pip install langchain-together` |
| Mistral | `langchain-mistralai` | `pip install langchain-mistralai` |
| HuggingFace | `langchain-huggingface` | `pip install langchain-huggingface` |
| Groq | `langchain-groq` | `pip install langchain-groq` |
| Ollama | `langchain-ollama` | `pip install langchain-ollama` |
| DeepSeek | `langchain-deepseek` | `pip install langchain-deepseek` |
| xAI | `langchain-xai` | `pip install langchain-xai` |
| Perplexity | `langchain-perplexity` | `pip install langchain-perplexity` |
| Upstage | `langchain-upstage` | `pip install langchain-upstage` |
| NVIDIA | `langchain-nvidia-ai-endpoints` | `pip install langchain-nvidia-ai-endpoints` |
| IBM Watsonx | `langchain-ibm` | `pip install langchain-ibm` |

== Credentials ==
The following environment variables may be required depending on your chosen provider:

* `OPENAI_API_KEY`: OpenAI API key
* `ANTHROPIC_API_KEY`: Anthropic API key
* `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
* `AZURE_OPENAI_API_BASE`: Azure OpenAI endpoint URL
* `AZURE_OPENAI_API_VERSION`: Azure API version
* `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud service account JSON
* `AWS_ACCESS_KEY_ID`: AWS access key for Bedrock
* `AWS_SECRET_ACCESS_KEY`: AWS secret key for Bedrock
* `COHERE_API_KEY`: Cohere API key
* `MISTRAL_API_KEY`: Mistral AI API key
* `GROQ_API_KEY`: Groq API key
* `HF_TOKEN`: HuggingFace API token
* `LANGSMITH_API_KEY`: Optional LangSmith API key for tracing
* `LANGSMITH_TRACING`: Set to "true" to enable LangSmith tracing

== Quick Install ==
<syntaxhighlight lang="bash">
# Install core packages
pip install langchain langchain-core langgraph pydantic

# Install provider package(s) as needed, e.g.:
pip install langchain-openai langchain-anthropic
</syntaxhighlight>

== Code Evidence ==

Provider package validation from `chat_models/base.py:533-537`:
<syntaxhighlight lang="python">
def _check_pkg(pkg: str, *, pkg_kebab: str | None = None) -> None:
    if not util.find_spec(pkg):
        pkg_kebab = pkg_kebab if pkg_kebab is not None else pkg.replace("_", "-")
        msg = f"Unable to import {pkg}. Please install with `pip install -U {pkg_kebab}`"
        raise ImportError(msg)
</syntaxhighlight>

Provider inference from model name in `chat_models/base.py:489-512`:
<syntaxhighlight lang="python">
def _attempt_infer_model_provider(model_name: str) -> str | None:
    if any(model_name.startswith(pre) for pre in ("gpt-", "o1", "o3")):
        return "openai"
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("command"):
        return "cohere"
    # ... more provider inference logic
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Unable to import langchain_openai. Please install with pip install -U langchain-openai` || Provider package not installed || `pip install langchain-openai`
|-
|| `Unable to infer model provider for {model}, please specify model_provider directly.` || Model name doesn't match known prefixes || Pass explicit `model_provider` parameter
|-
|| `ValueError: Unsupported model_provider` || Invalid provider name || Check supported providers list in docstring
|-
|| `AuthenticationError` / `Invalid API Key` || Missing or invalid API key || Set appropriate environment variable
|}

== Compatibility Notes ==

* '''Python Version:''' Requires Python 3.9+ for type hint features and async support
* '''Pydantic:''' Requires Pydantic v2; v1 is not supported
* '''Provider Packages:''' Each provider package may have additional dependencies (e.g., `tiktoken` for OpenAI token counting)
* '''Ollama:''' Requires running Ollama server locally; falls back to `langchain-community` if `langchain-ollama` unavailable

== Related Pages ==
* [[requires_env::Implementation:langchain-ai_langchain_init_chat_model]]
* [[requires_env::Implementation:langchain-ai_langchain_BaseTool_and_StructuredTool]]
* [[requires_env::Implementation:langchain-ai_langchain_AgentMiddleware_class]]
* [[requires_env::Implementation:langchain-ai_langchain_ResponseFormat_strategies]]
* [[requires_env::Implementation:langchain-ai_langchain_create_agent_graph_building]]
* [[requires_env::Implementation:langchain-ai_langchain_CompiledStateGraph_invocation]]
* [[requires_env::Implementation:langchain-ai_langchain_SchemaSpec_class]]
* [[requires_env::Implementation:langchain-ai_langchain_ResponseFormat_type_union]]
* [[requires_env::Implementation:langchain-ai_langchain_OutputToolBinding_class]]
* [[requires_env::Implementation:langchain-ai_langchain_agent_model_binding]]
* [[requires_env::Implementation:langchain-ai_langchain_parse_with_schema]]
* [[requires_env::Implementation:langchain-ai_langchain_structured_output_error_classes]]
* [[requires_env::Implementation:langchain-ai_langchain_model_parsing_functions]]
* [[requires_env::Implementation:langchain-ai_langchain_check_pkg]]
* [[requires_env::Implementation:langchain-ai_langchain_init_chat_model_helper]]
* [[requires_env::Implementation:langchain-ai_langchain_ConfigurableModel_class]]
* [[requires_env::Implementation:langchain-ai_langchain_ConfigurableModel_declarative_methods]]
