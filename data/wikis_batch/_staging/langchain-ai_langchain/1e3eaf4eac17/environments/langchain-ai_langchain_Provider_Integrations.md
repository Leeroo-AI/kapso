# Environment: Provider Integrations

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|Chat Models Base|libs/langchain_v1/langchain/chat_models/base.py]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::LLM_Providers]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

## Overview

Optional provider integration packages for connecting LangChain to various LLM providers (OpenAI, Anthropic, Google, AWS, etc.).

### Description

LangChain uses a modular architecture where each LLM provider has its own integration package. The core `init_chat_model` function dynamically loads provider packages based on the model being initialized. This allows users to install only the integrations they need, keeping dependencies minimal.

### Usage

Install provider packages based on which LLM providers you need to use. Each provider requires its own API key/credentials. The `init_chat_model` function will automatically detect missing packages and provide installation instructions.

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| Base || langchain-ai_langchain_Python_Runtime || Requires core environment
|-
| Network || Internet access || API calls to provider endpoints
|}

## Dependencies

### Provider Packages

Each provider has its own integration package:

{| class="wikitable"
! Provider !! Package !! Install Command
|-
| OpenAI || `langchain-openai` || `pip install langchain-openai`
|-
| Anthropic || `langchain-anthropic` || `pip install langchain-anthropic`
|-
| Azure OpenAI || `langchain-openai` || `pip install langchain-openai`
|-
| Azure AI || `langchain-azure-ai` || `pip install langchain-azure-ai`
|-
| Google VertexAI || `langchain-google-vertexai` || `pip install langchain-google-vertexai`
|-
| Google GenAI || `langchain-google-genai` || `pip install langchain-google-genai`
|-
| AWS Bedrock || `langchain-aws` || `pip install langchain-aws`
|-
| Cohere || `langchain-cohere` || `pip install langchain-cohere`
|-
| Fireworks || `langchain-fireworks` || `pip install langchain-fireworks`
|-
| Together || `langchain-together` || `pip install langchain-together`
|-
| MistralAI || `langchain-mistralai` || `pip install langchain-mistralai`
|-
| HuggingFace || `langchain-huggingface` || `pip install langchain-huggingface`
|-
| Groq || `langchain-groq` || `pip install langchain-groq`
|-
| Ollama || `langchain-ollama` || `pip install langchain-ollama`
|-
| DeepSeek || `langchain-deepseek` || `pip install langchain-deepseek`
|-
| xAI || `langchain-xai` || `pip install langchain-xai`
|-
| Perplexity || `langchain-perplexity` || `pip install langchain-perplexity`
|-
| Upstage || `langchain-upstage` || `pip install langchain-upstage`
|-
| IBM || `langchain-ibm` || `pip install langchain-ibm`
|-
| NVIDIA || `langchain-nvidia-ai-endpoints` || `pip install langchain-nvidia-ai-endpoints`
|}

## Credentials

Each provider requires its own API key. Set these as environment variables:

* `OPENAI_API_KEY`: OpenAI API token
* `ANTHROPIC_API_KEY`: Anthropic API token
* `GOOGLE_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS`: Google authentication
* `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: AWS Bedrock credentials
* `COHERE_API_KEY`: Cohere API token
* `FIREWORKS_API_KEY`: Fireworks API token
* `TOGETHER_API_KEY`: Together AI API token
* `MISTRAL_API_KEY`: Mistral AI API token
* `GROQ_API_KEY`: Groq API token
* `HUGGINGFACE_API_KEY`: HuggingFace API token

## Quick Install

```bash
# Install commonly used providers
pip install langchain-openai langchain-anthropic

# Install all optional providers (for development)
pip install langchain[openai,anthropic,google-vertexai,aws]
```

## Code Evidence

Provider package verification from `libs/langchain_v1/langchain/chat_models/base.py:533-540`:
```python
def _check_pkg(pkg: str, pkg_kebab: str | None = None) -> None:
    """Check if a package is installed and raise ImportError if not."""
    if not util.find_spec(pkg):
        pkg_kebab = pkg_kebab or pkg.replace("_", "-")
        raise ImportError(f"Unable to import {pkg}. Please install with: pip install {pkg_kebab}")
```

Provider instantiation from `libs/langchain_v1/langchain/chat_models/base.py:339-350`:
```python
model, model_provider = _parse_model(model, model_provider)
if model_provider == "openai":
    _check_pkg("langchain_openai")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, **kwargs)
if model_provider == "anthropic":
    _check_pkg("langchain_anthropic")
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, **kwargs)
```

Model prefix inference from `libs/langchain_v1/langchain/chat_models/base.py:93-106`:
```python
# The following providers will be inferred based on these model prefixes:
# - `gpt-...` | `o1...` | `o3...`       -> `openai`
# - `claude...`                         -> `anthropic`
# - `amazon...`                         -> `bedrock`
# - `gemini...`                         -> `google_vertexai`
# - `command...`                        -> `cohere`
# - `accounts/fireworks...`             -> `fireworks`
# - `mistral...`                        -> `mistralai`
# - `deepseek...`                       -> `deepseek`
# - `grok...`                           -> `xai`
```

## Common Errors

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: Unable to import langchain_openai` || OpenAI package not installed || `pip install langchain-openai`
|-
|| `ValueError: model_provider cannot be inferred` || Unknown model prefix || Specify `model_provider` explicitly
|-
|| `AuthenticationError: Invalid API key` || Missing or invalid credentials || Set the appropriate API key environment variable
|}

## Compatibility Notes

* **Model Prefix Inference:** Model names starting with certain prefixes (e.g., `gpt-`, `claude-`) automatically infer the provider
* **Custom Base URLs:** Most providers support `base_url` parameter for custom endpoints
* **Rate Limiting:** Use `rate_limiter` parameter for controlling request rates

## Related Pages

* [[required_by::Implementation:langchain-ai_langchain_init_chat_model]]
* [[required_by::Implementation:langchain-ai_langchain_init_chat_model_helper]]
* [[required_by::Implementation:langchain-ai_langchain_check_pkg]]
