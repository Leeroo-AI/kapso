# Client Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|OpenAI Python Client|https://github.com/openai/openai-python]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::API_Client]], [[domain::Integration]]
|-
! Last Updated
| [[last_updated::2025-01-15 14:00 GMT]]
|}

== Overview ==

Python environment with the OpenAI Python client library for consuming vLLM's OpenAI-compatible API endpoints.

=== Description ===

This environment provides the client-side context for interacting with a vLLM server via its OpenAI-compatible HTTP API. It uses the official OpenAI Python client library (`openai>=1.0.0`) to send requests and process responses. The client can be used for chat completions, text completions, embeddings, and tool calling workflows.

=== Usage ===

Use this environment when **consuming vLLM's API** from client code:
- Initializing OpenAI client with vLLM server URL
- Sending chat completion requests
- Processing streaming and non-streaming responses
- Handling tool calls and function calling
- Multimodal inputs (images, videos) via API

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Any (Linux, macOS, Windows) || Cross-platform support
|-
| Python || Python 3.8+ || openai package requirements
|-
| Network || HTTP/HTTPS access to vLLM server || Default port 8000
|-
| Memory || 1GB+ RAM || Minimal for client operations
|}

== Dependencies ==

=== Python Packages ===
* `openai` >= 1.0.0 (official OpenAI client library)
* `httpx` >= 0.24.0 (HTTP client, dependency of openai)
* `pydantic` >= 2.0.0 (data validation)

=== Optional Dependencies ===
* `aiohttp` (for async HTTP operations)
* `requests` (alternative HTTP client)

== Credentials ==

The following may be required:
* `api_key`: API key if server has authentication enabled (use "EMPTY" if no auth)
* `base_url`: vLLM server URL (e.g., `http://localhost:8000/v1`)

== Quick Install ==

<syntaxhighlight lang="bash">
# Install OpenAI client
pip install openai>=1.0.0

# Verify installation
python -c "from openai import OpenAI; print('OpenAI client available')"
</syntaxhighlight>

== Code Evidence ==

Client initialization pattern from `examples/online_serving/openai_chat_completion_client.py`:
<syntaxhighlight lang="python">
from openai import OpenAI

# Initialize client pointing to vLLM server
client = OpenAI(
    api_key="EMPTY",  # Use "EMPTY" if no authentication
    base_url="http://localhost:8000/v1",  # vLLM server URL
)
</syntaxhighlight>

Chat completion request from `examples/online_serving/openai_chat_completion_client.py`:
<syntaxhighlight lang="python">
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",  # Model name as served by vLLM
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    temperature=0.7,
    max_tokens=100,
)
print(response.choices[0].message.content)
</syntaxhighlight>

Streaming response handling:
<syntaxhighlight lang="python">
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Connection refused` || vLLM server not running || Start server with `vllm serve <model>`
|-
|| `401 Unauthorized` || API key required but not provided || Set correct `api_key` in client initialization
|-
|| `404 Not Found` || Incorrect base_url || Ensure base_url ends with `/v1`
|-
|| `Model not found` || Model name mismatch || Use exact model name as served (check `/v1/models`)
|-
|| `openai.APITimeoutError` || Request timeout || Increase timeout or reduce request complexity
|}

== Compatibility Notes ==

* '''OpenAI SDK 1.0+:''' Required; older versions have different API signatures.
* '''Sync vs Async:''' Use `OpenAI()` for sync, `AsyncOpenAI()` for async operations.
* '''Streaming:''' Set `stream=True` for real-time token delivery.
* '''Tool Calling:''' vLLM supports OpenAI's function calling format; some models require specific chat templates.
* '''Multimodal:''' Image inputs supported via base64 or URL in message content.

== Usage Examples ==

=== Basic Chat ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
print(response.choices[0].message.content)
</syntaxhighlight>

=== Streaming Response ===
<syntaxhighlight lang="python">
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "Count to 10."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
</syntaxhighlight>

=== With Tools/Functions ===
<syntaxhighlight lang="python">
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }
}]
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=tools,
)
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Implementation:vllm-project_vllm_OpenAI_client_init]]
* [[requires_env::Implementation:vllm-project_vllm_chat_completions_create]]
* [[requires_env::Implementation:vllm-project_vllm_ChatCompletion_processing]]
