{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Chat Models|https://docs.langchain.com/oss/python/langchain/overview]]
* [[source::Blog|LangChain Provider Integrations|https://docs.langchain.com/oss/python/integrations/providers]]
|-
! Domains
| [[domain::LLM]], [[domain::Chat_Models]], [[domain::Provider_Abstraction]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Design pattern that abstracts chat model initialization across multiple LLM providers through a unified factory interface.

=== Description ===

Chat Model Initialization is an abstraction layer that decouples application code from specific LLM provider implementations. Instead of directly instantiating provider-specific classes (ChatOpenAI, ChatAnthropic, etc.), a unified factory function handles provider detection, package validation, and model instantiation.

This pattern solves several critical problems:
* **Provider Lock-in:** Applications can switch providers without code changes
* **Configuration Complexity:** Provider-specific initialization details are encapsulated
* **Runtime Flexibility:** Models can be configured/switched at runtime via config dictionaries
* **Consistent Interface:** All models expose the same `invoke`, `stream`, `batch` interface regardless of provider

=== Usage ===

Use this principle when:
* Building LLM applications that may need to switch between providers
* Creating configurable pipelines where model choice is determined at runtime
* Developing multi-model systems that need consistent interfaces
* Abstracting provider details from business logic

Key decision factors:
* Need provider flexibility? → Use factory pattern
* Single fixed provider? → Direct instantiation may be simpler
* Runtime model switching? → Use configurable model pattern

== Theoretical Basis ==

The Chat Model Initialization principle implements the **Abstract Factory Pattern** from object-oriented design, combined with **Strategy Pattern** for runtime configurability.

'''Core Components:'''

1. **Provider Detection:** Map model name prefixes to providers
<syntaxhighlight lang="python">
# Pseudo-code for provider inference
PREFIX_TO_PROVIDER = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "claude": "anthropic",
    "gemini": "google_vertexai",
    "command": "cohere",
    ...
}

def infer_provider(model_name: str) -> str:
    for prefix, provider in PREFIX_TO_PROVIDER.items():
        if model_name.startswith(prefix):
            return provider
    raise ValueError("Cannot infer provider")
</syntaxhighlight>

2. **Package Validation:** Ensure provider integration is installed
<syntaxhighlight lang="python">
# Pseudo-code for package validation
def check_package(provider: str) -> None:
    pkg = f"langchain_{provider}"
    if not importlib.find_spec(pkg):
        raise ImportError(f"Install {pkg} to use {provider} models")
</syntaxhighlight>

3. **Instantiation Dispatch:** Route to correct provider class
<syntaxhighlight lang="python">
# Pseudo-code for instantiation
PROVIDER_CLASSES = {
    "openai": ("langchain_openai", "ChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    ...
}

def instantiate(provider: str, model: str, **kwargs):
    module, cls_name = PROVIDER_CLASSES[provider]
    cls = getattr(import_module(module), cls_name)
    return cls(model=model, **kwargs)
</syntaxhighlight>

4. **Configurable Wrapper:** Enable runtime configuration
<syntaxhighlight lang="python">
# Pseudo-code for configurable model
class ConfigurableModel:
    def __init__(self, default_config, configurable_fields):
        self.default_config = default_config
        self.configurable_fields = configurable_fields

    def invoke(self, input, config=None):
        # Merge runtime config with defaults
        params = {**self.default_config, **extract_config(config)}
        # Instantiate actual model
        model = instantiate(**params)
        return model.invoke(input)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_init_chat_model]]

=== Used By Workflows ===
* Agent_Creation_Workflow (Step 1)
* Chat_Model_Initialization_Workflow (Step 3)
