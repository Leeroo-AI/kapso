{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|OpenAI Python Library|https://github.com/openai/openai-python]]
|-
! Domains
| [[domain::NLP]], [[domain::Client_SDK]], [[domain::API_Integration]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Wrapper documentation for initializing the OpenAI Python client to connect to a vLLM server endpoint.

=== Description ===

The OpenAI Python SDK (`openai` package) can be used as-is to communicate with vLLM servers because vLLM implements an OpenAI-compatible API. The key configuration is setting `base_url` to point to the vLLM server instead of the official OpenAI endpoint.

This wrapper documentation covers vLLM-specific usage patterns and common configurations.

=== Usage ===

Use the OpenAI client with vLLM when:
- You want to use familiar OpenAI SDK patterns
- Migrating existing OpenAI-based applications to vLLM
- Building applications that can switch between OpenAI and vLLM
- Testing prompts locally before deploying to OpenAI

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/openai/openai-python openai (PyPI)]
* '''vLLM Server:''' vllm/entrypoints/openai/api_server.py

=== Signature ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(
    api_key: str = "EMPTY",           # API key (use "EMPTY" for no auth)
    base_url: str = "http://localhost:8000/v1",  # vLLM server URL
    timeout: float = 60.0,            # Request timeout in seconds
    max_retries: int = 2,             # Number of retries
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from openai import OpenAI
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| api_key || str || No || API key for authentication (use "EMPTY" for no auth)
|-
| base_url || str || Yes || vLLM server URL with /v1 suffix
|-
| timeout || float || No || Request timeout in seconds (default: 60)
|-
| max_retries || int || No || Retry count for failed requests (default: 2)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| client || OpenAI || Configured OpenAI client pointing to vLLM server
|}

== Usage Examples ==

=== Basic Client Setup ===
<syntaxhighlight lang="python">
from openai import OpenAI

# Connect to local vLLM server (no authentication)
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# Verify connection
models = client.models.list()
print(f"Available models: {[m.id for m in models.data]}")
</syntaxhighlight>

=== With API Key Authentication ===
<syntaxhighlight lang="python">
from openai import OpenAI

# Connect to authenticated vLLM server
client = OpenAI(
    api_key="sk-vllm-secret-key",  # Must match server's --api-key
    base_url="http://localhost:8000/v1",
)

# API calls now include authentication header
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
</syntaxhighlight>

=== Environment Variable Configuration ===
<syntaxhighlight lang="python">
import os
from openai import OpenAI

# Set environment variables (recommended for production)
os.environ["OPENAI_API_KEY"] = "sk-vllm-key"
os.environ["OPENAI_BASE_URL"] = "http://vllm-server:8000/v1"

# Client reads from environment automatically
client = OpenAI()  # No explicit configuration needed
</syntaxhighlight>

=== Remote Server Connection ===
<syntaxhighlight lang="python">
from openai import OpenAI

# Connect to remote vLLM deployment
client = OpenAI(
    api_key="production-api-key",
    base_url="https://vllm.example.com/v1",
    timeout=120.0,  # Longer timeout for large requests
    max_retries=3,
)
</syntaxhighlight>

=== Async Client ===
<syntaxhighlight lang="python">
from openai import AsyncOpenAI
import asyncio

# Async client for concurrent requests
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

async def generate(prompt):
    response = await client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Run multiple requests concurrently
async def main():
    prompts = ["Hello!", "What is AI?", "Write a haiku."]
    results = await asyncio.gather(*[generate(p) for p in prompts])
    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}\nA: {result}\n")

asyncio.run(main())
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_OpenAI_Client_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Client_Environment]]
