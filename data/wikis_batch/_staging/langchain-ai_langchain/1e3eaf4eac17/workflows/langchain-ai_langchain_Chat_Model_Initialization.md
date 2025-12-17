# Workflow: Chat Model Initialization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Models|https://docs.langchain.com/oss/python/langchain/models]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Integration]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Provider-agnostic process for initializing chat models from any supported LLM provider using a unified interface.

=== Description ===
This workflow covers the use of `init_chat_model()` to create chat model instances from a wide variety of providers (OpenAI, Anthropic, Google, AWS Bedrock, etc.) using a single, consistent API. The function supports both fixed models (specify upfront) and configurable models (choose at runtime). It handles provider inference, package verification, and model instantiation automatically.

=== Usage ===
Execute this workflow when you need to:
* Create a chat model without coupling to a specific provider
* Build applications that can switch between providers at runtime
* Support multiple model backends in the same application
* Prototype quickly without worrying about provider-specific APIs

== Execution Steps ==

=== Step 1: Parse Model Identifier ===
[[step::Principle:langchain-ai_langchain_Model_String_Parsing]]

Parse the model string to extract the provider and model name. The format supports explicit provider specification (`openai:gpt-4`) or automatic inference from model name prefixes (e.g., `gpt-*` maps to OpenAI, `claude*` maps to Anthropic).

'''Key considerations:'''
* Explicit format: `{provider}:{model_name}` takes precedence
* Inference rules cover common patterns: gpt-/o1/o3 (OpenAI), claude (Anthropic), gemini (Google), etc.
* Provider names are normalized (hyphens to underscores, lowercase)

=== Step 2: Verify Provider Package ===
[[step::Principle:langchain-ai_langchain_Provider_Package_Verification]]

Check that the required integration package is installed for the selected provider. Each provider has its own LangChain integration package (e.g., `langchain-openai`, `langchain-anthropic`) that must be available.

'''Key considerations:'''
* Package names follow pattern `langchain-{provider}`
* Clear error messages indicate which package to install
* Some providers have fallback packages (e.g., Ollama)

=== Step 3: Instantiate Provider Model ===
[[step::Principle:langchain-ai_langchain_Provider_Model_Instantiation]]

Create the provider-specific chat model instance by importing the appropriate class and passing the model name along with any additional configuration kwargs.

'''What happens:'''
* Provider-specific class is imported (e.g., `ChatOpenAI`, `ChatAnthropic`)
* Model is instantiated with the parsed model name
* Additional kwargs (temperature, max_tokens, timeout, etc.) are forwarded

=== Step 4: Configure Declarative Operations (Optional) ===
[[step::Principle:langchain-ai_langchain_Model_Declarative_Operations]]

For configurable models, queue up declarative operations like `bind_tools` and `with_structured_output` that will be applied when the model is actually instantiated at runtime.

'''Key considerations:'''
* Operations are stored and replayed on each model instantiation
* Allows tool binding before knowing which provider will be used
* Configuration can be overridden per-invocation via config

== Execution Diagram ==
{{#mermaid:graph TD
    A[Parse Model Identifier] --> B[Verify Provider Package]
    B --> C[Instantiate Provider Model]
    C --> D[Configure Declarative Operations]
}}

== Related Pages ==
* [[step::Principle:langchain-ai_langchain_Model_String_Parsing]]
* [[step::Principle:langchain-ai_langchain_Provider_Package_Verification]]
* [[step::Principle:langchain-ai_langchain_Provider_Model_Instantiation]]
* [[step::Principle:langchain-ai_langchain_Model_Declarative_Operations]]
