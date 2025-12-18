{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/models]]
|-
! Domains
| [[domain::LLMs]], [[domain::Provider_Abstraction]], [[domain::Chat_Models]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for initializing chat models from any supported provider using LangChain's unified `init_chat_model` factory interface.

=== Description ===
This workflow covers the model initialization process that abstracts away provider-specific details. The `init_chat_model` factory function provides a single interface to instantiate chat models from 20+ providers, with automatic provider inference from model names and optional runtime configurability.

Key capabilities:
* Unified interface across OpenAI, Anthropic, Google, Bedrock, and 15+ other providers
* Automatic provider inference from model name prefixes
* Runtime-configurable models for dynamic provider/model switching
* Declarative operation queuing (bind_tools, with_structured_output)
* Type-safe parameter handling per provider

=== Usage ===
Execute this workflow when you need to:
* Initialize a chat model without provider-specific imports
* Build applications that support multiple model providers
* Create runtime-configurable model selections
* Bind tools or structured output schemas to models

Typical scenarios: multi-provider chatbots, A/B testing models, user-selectable model preferences.

== Execution Steps ==

=== Step 1: Model Identifier Parsing ===
[[step::Principle:langchain-ai_langchain_Model_Identifier_Parsing]]

Parse the model identifier to determine the model name and provider. The identifier can be a simple model name (with provider inference) or use explicit `provider:model` syntax.

'''Supported formats:'''
* Explicit: `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-5-20250929"`
* Inferred: `"gpt-4o"` → openai, `"claude-3-opus"` → anthropic, `"gemini-pro"` → google_vertexai

'''Provider inference rules:'''
* `gpt-*`, `o1*`, `o3*` → openai
* `claude*` → anthropic
* `gemini*` → google_vertexai
* `command*` → cohere
* `deepseek*` → deepseek

=== Step 2: Provider Package Validation ===
[[step::Principle:langchain-ai_langchain_Provider_Package_Validation]]

Validate that the required integration package is installed for the target provider. Each provider requires its own `langchain-{provider}` package with the specific model implementation.

'''Package mapping:'''
* openai, azure_openai → `langchain-openai`
* anthropic → `langchain-anthropic`
* google_vertexai, google_anthropic_vertex → `langchain-google-vertexai`
* bedrock, bedrock_converse → `langchain-aws`
* ollama → `langchain-ollama`

=== Step 3: Model Instantiation ===
[[step::Principle:langchain-ai_langchain_Model_Instantiation]]

Create the provider-specific chat model instance with the given parameters. The factory maps the universal kwargs to the provider's `__init__` signature and constructs the model.

'''Common parameters:'''
* `temperature`: Controls randomness (0.0-1.0)
* `max_tokens`: Maximum output token count
* `timeout`: API timeout in seconds
* `max_retries`: Retry attempts on transient failures
* `base_url`: Custom API endpoint (self-hosted models)
* `rate_limiter`: BaseRateLimiter instance for rate control

=== Step 4: Configurable Model Setup (Optional) ===
[[step::Principle:langchain-ai_langchain_Configurable_Model_Setup]]

Optionally wrap the model in a configurable interface that allows runtime parameter changes. This enables dynamic model switching without code changes, useful for A/B testing or user preferences.

'''Configuration modes:'''
* Fixed model: No `configurable_fields` - returns direct BaseChatModel
* Partially configurable: Specify which fields can change at runtime
* Fully configurable: `configurable_fields="any"` - all fields changeable

'''Security note:''' Full configurability (`"any"`) exposes sensitive fields like `api_key` and `base_url` to runtime changes.

=== Step 5: Declarative Operation Binding ===
[[step::Principle:langchain-ai_langchain_Declarative_Operation_Binding]]

Apply declarative operations like tool binding or structured output configuration. For configurable models, operations are queued and applied when the actual model is instantiated at runtime.

'''Supported operations:'''
* `bind_tools()`: Attach tool schemas for function calling
* `with_structured_output()`: Configure JSON schema responses
* `with_config()`: Set runtime configuration defaults

== Execution Diagram ==
{{#mermaid:graph TD
    A[Model Identifier Parsing] --> B[Provider Package Validation]
    B --> C{Package Installed?}
    C -->|No| D[ImportError]
    C -->|Yes| E[Model Instantiation]
    E --> F{Configurable?}
    F -->|Yes| G[Configurable Model Setup]
    F -->|No| H[Direct BaseChatModel]
    G --> I[Declarative Operation Binding]
    H --> I
    I --> J[Return Model]
}}

== Related Pages ==
* [[step::Principle:langchain-ai_langchain_Model_Identifier_Parsing]]
* [[step::Principle:langchain-ai_langchain_Provider_Package_Validation]]
* [[step::Principle:langchain-ai_langchain_Model_Instantiation]]
* [[step::Principle:langchain-ai_langchain_Configurable_Model_Setup]]
* [[step::Principle:langchain-ai_langchain_Declarative_Operation_Binding]]
