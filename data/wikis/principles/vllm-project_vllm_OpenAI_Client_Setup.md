{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Client_SDK]], [[domain::API_Integration]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of configuring an HTTP client to communicate with an OpenAI-compatible LLM serving endpoint.

=== Description ===

OpenAI Client Setup establishes the connection between a client application and an LLM server. Because vLLM implements the OpenAI API specification, standard OpenAI client libraries can be used with minimal configuration changes.

Key aspects include:
1. **Endpoint Configuration:** Pointing the client to the correct server URL
2. **Authentication:** Providing API keys when server requires them
3. **Connection Settings:** Timeouts, retries, and connection pooling
4. **Protocol Selection:** HTTP vs HTTPS, sync vs async

This enables applications to switch between OpenAI and vLLM backends without code changes beyond configuration.

=== Usage ===

Set up OpenAI clients when:
- Building applications that consume LLM APIs
- Creating test harnesses for prompt development
- Migrating applications between providers
- Implementing retry and error handling logic
- Managing multiple LLM backends

== Theoretical Basis ==

'''Client Architecture:'''

<syntaxhighlight lang="python">
# Conceptual client structure
class LLMClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.session = create_http_session()

    def _request(self, method, endpoint, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}{endpoint}"
        return self.session.request(method, url, headers=headers, **kwargs)
</syntaxhighlight>

'''OpenAI API Compatibility:'''

vLLM implements these OpenAI endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| /v1/models | GET | List available models |
| /v1/completions | POST | Text completion (legacy) |
| /v1/chat/completions | POST | Chat completion |
| /v1/embeddings | POST | Text embeddings |

'''Connection Configuration:'''

<syntaxhighlight lang="python">
# Key configuration parameters
config = {
    "base_url": "http://server:8000/v1",  # Must end with /v1
    "api_key": "key",                      # Or "EMPTY" for no auth
    "timeout": 60,                         # Request timeout
    "max_retries": 3,                      # Automatic retries
}
</syntaxhighlight>

'''Error Handling:'''

<syntaxhighlight lang="python">
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

try:
    response = client.chat.completions.create(...)
except APITimeoutError:
    # Server took too long to respond
    pass
except RateLimitError:
    # Too many requests (implement backoff)
    pass
except APIError as e:
    # Other API errors
    print(f"Status: {e.status_code}, Message: {e.message}")
</syntaxhighlight>

'''Environment-Based Configuration:'''

For production deployments, use environment variables:

<syntaxhighlight lang="bash">
# Client automatically reads these
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://vllm.example.com/v1"
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_OpenAI_client_init]]
