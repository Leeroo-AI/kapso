# OpenAI Client (Wrapper)

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Python SDK|https://github.com/openai/openai-python]]
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Serving]], [[domain::Client]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Usage pattern for the OpenAI Python SDK configured to communicate with vLLM's OpenAI-compatible server.

=== Description ===

The official `openai` Python package can connect to vLLM servers by configuring the `base_url` parameter. This enables drop-in replacement of OpenAI's API with a self-hosted vLLM server.

This is a **Wrapper Doc** documenting how to use an external library (OpenAI SDK) with vLLM.

=== Usage ===

Use the OpenAI client with vLLM when:
* Building applications that need OpenAI API compatibility
* Migrating existing OpenAI-based code to vLLM
* Using libraries that expect OpenAI API format (LangChain, LlamaIndex)

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/openai/openai-python OpenAI Python SDK]
* '''vLLM Server:''' vllm/entrypoints/openai/

=== Signature ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(
    base_url: str = "https://api.openai.com/v1",  # Override for vLLM
    api_key: str = ...,                            # Required (can be placeholder)
    timeout: float | Timeout | None = NOT_GIVEN,   # Request timeout
    max_retries: int = 2,                          # Retry count
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from openai import OpenAI
# pip install openai
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_url || str || Yes (for vLLM) || vLLM server URL (e.g., "http://localhost:8000/v1")
|-
| api_key || str || Yes || API key (use "EMPTY" if server has no auth)
|-
| timeout || float || No || Request timeout in seconds
|-
| max_retries || int || No || Number of retries (default: 2)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| client || OpenAI || Configured client instance
|-
| client.chat.completions || Resource || Chat completion methods
|-
| client.completions || Resource || Legacy completion methods
|}

== Usage Examples ==

=== Basic Client Setup ===
<syntaxhighlight lang="python">
from openai import OpenAI

# Connect to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # vLLM doesn't require real key by default
)

# Test connection
models = client.models.list()
print(f"Available models: {[m.id for m in models.data]}")
</syntaxhighlight>

=== With API Key Authentication ===
<syntaxhighlight lang="python">
from openai import OpenAI

# Server started with: vllm serve ... --api-key "sk-secret"
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-secret",  # Must match server's --api-key
)
</syntaxhighlight>

=== Chat Completion Request ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
</syntaxhighlight>

=== Streaming Response ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Write a poem about AI."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
</syntaxhighlight>

=== With Timeout and Retries ===
<syntaxhighlight lang="python">
from openai import OpenAI
import httpx

# Configure for long-running requests
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    timeout=httpx.Timeout(300.0, connect=10.0),  # 5 min total, 10s connect
    max_retries=3,
)
</syntaxhighlight>

=== LangChain Integration ===
<syntaxhighlight lang="python">
from langchain_openai import ChatOpenAI

# LangChain uses OpenAI SDK under the hood
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

response = llm.invoke("What is machine learning?")
print(response.content)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_API_Client_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
