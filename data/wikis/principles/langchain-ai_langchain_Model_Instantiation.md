{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|LangChain Integrations|https://docs.langchain.com/oss/python/integrations/providers]]
* [[source::Doc|Factory Pattern|https://refactoring.guru/design-patterns/factory-method]]
|-
! Domains
| [[domain::LLM]], [[domain::Provider_Abstraction]], [[domain::Design_Patterns]]
|-
! Last Updated
| [[last_updated::2024-12-18 14:00 GMT]]
|}

== Overview ==

Factory pattern that creates provider-specific chat model instances from a unified interface.

=== Description ===

Model Instantiation is the process of creating concrete chat model objects based on provider and configuration. This pattern:
* Encapsulates provider-specific class knowledge
* Provides a single entry point for all providers
* Handles provider-specific initialization requirements
* Returns a common `BaseChatModel` interface

The factory approach means application code never needs to import or know about `ChatOpenAI`, `ChatAnthropic`, etc.—it just works with `BaseChatModel`.

=== Usage ===

Model Instantiation is the core operation when:
* Starting any LLM-powered application
* Switching between providers in A/B tests
* Building multi-model systems
* Creating model pools for load balancing

== Theoretical Basis ==

Model Instantiation implements the **Factory Method** pattern.

'''1. Provider Dispatch Logic'''

<syntaxhighlight lang="python">
# Pseudo-code for instantiation dispatch
def instantiate_model(provider: str, model: str, **kwargs) -> BaseChatModel:
    """Factory method for model instantiation."""

    if provider == "openai":
        check_pkg("langchain_openai")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, **kwargs)

    elif provider == "anthropic":
        check_pkg("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, **kwargs)

    elif provider == "bedrock":
        check_pkg("langchain_aws")
        from langchain_aws import ChatBedrock
        # Note: Bedrock uses model_id instead of model
        return ChatBedrock(model_id=model, **kwargs)

    # ... other providers

    raise ValueError(f"Unsupported provider: {provider}")
</syntaxhighlight>

'''2. Common Interface Guarantee'''

<syntaxhighlight lang="python">
# All models expose the same interface
from langchain_core.language_models import BaseChatModel

def use_any_model(model: BaseChatModel, message: str):
    """Works with any provider's model."""
    # Common methods work on all models:
    response = model.invoke(message)          # Sync invocation
    async_response = await model.ainvoke(message)  # Async
    for chunk in model.stream(message):       # Streaming
        print(chunk.content)
    bound = model.bind_tools([tool])          # Tool binding
    structured = model.with_structured_output(Schema)  # Structured output
</syntaxhighlight>

'''3. Provider-Specific Parameters'''

<syntaxhighlight lang="python">
# Different providers accept different kwargs
PROVIDER_KWARGS = {
    "openai": {"api_key", "base_url", "organization", "timeout", "max_retries"},
    "anthropic": {"api_key", "base_url", "max_tokens"},
    "bedrock": {"region_name", "credentials_profile_name", "aws_access_key_id"},
    "ollama": {"base_url", "num_ctx", "num_gpu"},
}

# Factory passes kwargs through to provider class
model = init_chat_model(
    "gpt-4o",
    temperature=0,        # Common param
    api_key="sk-...",     # Provider-specific
    organization="org-..." # Provider-specific
)
</syntaxhighlight>

'''4. Lazy Import Pattern'''

<syntaxhighlight lang="python">
# Provider packages imported only when needed
# Benefits:
# 1. Faster startup (no unused imports)
# 2. Works with missing packages (until you use them)
# 3. Clearer errors (at point of use)

def instantiate_openai(model, **kwargs):
    # Import happens here, not at module load
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, **kwargs)
</syntaxhighlight>

'''5. Model Parameter Normalization'''

<syntaxhighlight lang="python">
# Some providers use different parameter names
def normalize_params(provider: str, **kwargs) -> dict:
    normalized = dict(kwargs)

    # model vs model_id
    if provider in {"bedrock", "ibm"}:
        if "model" in normalized:
            normalized["model_id"] = normalized.pop("model")

    # HuggingFace uses from_model_id factory
    # Handled specially in instantiation

    return normalized
</syntaxhighlight>

'''6. Error Handling'''

<syntaxhighlight lang="python">
def safe_instantiate(provider: str, model: str, **kwargs):
    """Instantiate with comprehensive error handling."""
    try:
        return instantiate_model(provider, model, **kwargs)
    except ImportError as e:
        raise ImportError(
            f"Provider '{provider}' requires additional package. "
            f"Install with: pip install langchain-{provider}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {provider}:{model}. "
            f"Check credentials and configuration. Error: {e}"
        ) from e
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:langchain-ai_langchain_init_chat_model_helper]]

=== Used By Workflows ===
* Chat_Model_Initialization_Workflow (Step 3)
