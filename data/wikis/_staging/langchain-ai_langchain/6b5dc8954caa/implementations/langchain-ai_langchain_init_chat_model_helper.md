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

Concrete tool for instantiating provider-specific chat model classes based on resolved provider name, provided by LangChain's chat model factory.

=== Description ===

`_init_chat_model_helper` is the internal function that performs the actual model instantiation. After provider resolution, this function:
* Maps provider name to the appropriate class
* Validates and imports the provider package
* Instantiates the model with provided kwargs
* Returns a ready-to-use `BaseChatModel` instance

This function encapsulates the provider-specific knowledge of which class to use for each provider.

=== Usage ===

Use this function (indirectly via `init_chat_model`) when:
* You need a fixed (non-configurable) model instance
* Initializing models for production use
* Building simple agent pipelines

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain]
* '''File:''' libs/langchain_v1/langchain/chat_models/base.py
* '''Lines:''' L332-461

=== Signature ===
<syntaxhighlight lang="python">
def _init_chat_model_helper(
    model: str,
    *,
    model_provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Initialize a chat model for a specific provider.

    Args:
        model: The model name/ID
        model_provider: The provider name (resolved by _parse_model if None)
        **kwargs: Provider-specific model kwargs (temperature, max_tokens, etc.)

    Returns:
        BaseChatModel instance ready for invocation

    Raises:
        ValueError: If provider is not supported
        ImportError: If provider package is not installed
    """
</syntaxhighlight>

=== Provider Mapping ===
<syntaxhighlight lang="python">
# Internal mapping (conceptual)
PROVIDER_MAP = {
    "openai": ("langchain_openai", "ChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI"),
    "azure_ai": ("langchain_azure_ai.chat_models", "AzureAIChatCompletionsModel"),
    "cohere": ("langchain_cohere", "ChatCohere"),
    "google_vertexai": ("langchain_google_vertexai", "ChatVertexAI"),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    "fireworks": ("langchain_fireworks", "ChatFireworks"),
    "ollama": ("langchain_ollama", "ChatOllama"),
    "together": ("langchain_together", "ChatTogether"),
    "mistralai": ("langchain_mistralai", "ChatMistralAI"),
    "huggingface": ("langchain_huggingface", "ChatHuggingFace"),
    "groq": ("langchain_groq", "ChatGroq"),
    "bedrock": ("langchain_aws", "ChatBedrock"),
    "bedrock_converse": ("langchain_aws", "ChatBedrockConverse"),
    "google_anthropic_vertex": ("langchain_google_vertexai.model_garden", "ChatAnthropicVertex"),
    "deepseek": ("langchain_deepseek", "ChatDeepSeek"),
    "nvidia": ("langchain_nvidia_ai_endpoints", "ChatNVIDIA"),
    "ibm": ("langchain_ibm", "ChatWatsonx"),
    "xai": ("langchain_xai", "ChatXAI"),
    "perplexity": ("langchain_perplexity", "ChatPerplexity"),
    "upstage": ("langchain_upstage", "ChatUpstage"),
}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal function - use via init_chat_model
from langchain.chat_models import init_chat_model

# The helper is called internally when configurable_fields is None
model = init_chat_model("gpt-4o")  # Calls _init_chat_model_helper
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || Model name/ID (e.g., "gpt-4o")
|-
| model_provider || str | None || No || Explicit provider (parsed from model if None)
|-
| **kwargs || Any || No || Provider-specific kwargs (temperature, max_tokens, api_key, etc.)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || BaseChatModel || Instantiated chat model ready for use
|}

== Usage Examples ==

=== Direct Model Instantiation ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# OpenAI
gpt4 = init_chat_model("gpt-4o", temperature=0)
response = gpt4.invoke("Hello!")

# Anthropic
claude = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.7, max_tokens=1000)
response = claude.invoke("Explain quantum computing")

# Google Vertex AI
gemini = init_chat_model("gemini-pro")
response = gemini.invoke("What's the weather like?")
</syntaxhighlight>

=== With Provider-Specific Options ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

# OpenAI with API key and base URL
model = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    api_key="sk-...",
    base_url="https://custom-endpoint.com",
    temperature=0,
    max_tokens=500,
)

# AWS Bedrock with region
bedrock = init_chat_model(
    "anthropic.claude-3-sonnet-20240229-v1:0",
    model_provider="bedrock",
    region_name="us-east-1",
)

# Ollama local model
ollama = init_chat_model(
    "llama3",
    model_provider="ollama",
    base_url="http://localhost:11434",
)
</syntaxhighlight>

=== Streaming and Async ===
<syntaxhighlight lang="python">
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o", streaming=True)

# Streaming
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)

# Async
import asyncio


async def main():
    model = init_chat_model("gpt-4o")
    response = await model.ainvoke("Hello!")
    print(response.content)


asyncio.run(main())
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:langchain-ai_langchain_Model_Instantiation]]

=== Requires Environment ===
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
