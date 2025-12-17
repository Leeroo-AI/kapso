# chat.completions.create (Wrapper)

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Chat API|https://platform.openai.com/docs/api-reference/chat/create]]
* [[source::Doc|OpenAI Python SDK|https://github.com/openai/openai-python]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Serving]], [[domain::API]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Usage pattern for the OpenAI SDK's chat completions method configured for vLLM's API endpoint.

=== Description ===

This is a **Wrapper Doc** documenting how to use the OpenAI SDK's `client.chat.completions.create()` method with vLLM. The method sends chat messages to vLLM and returns generated completions.

vLLM supports most OpenAI chat completions parameters including temperature, top_p, max_tokens, stop sequences, and streaming.

=== Usage ===

Use `chat.completions.create()` for:
* Single-turn and multi-turn conversations
* Streaming and non-streaming responses
* Structured output with JSON schemas
* Function/tool calling patterns

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/openai/openai-python OpenAI Python SDK]
* '''vLLM Endpoint:''' POST /v1/chat/completions

=== Signature ===
<syntaxhighlight lang="python">
def create(
    self,
    *,
    messages: list[ChatCompletionMessageParam],
    model: str,
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    max_tokens: int | None = None,
    n: int | None = None,
    presence_penalty: float | None = None,
    stop: str | list[str] | None = None,
    stream: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    tools: list[ChatCompletionToolParam] | None = None,
    **kwargs,
) -> ChatCompletion | Stream[ChatCompletionChunk]:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
# Use: client.chat.completions.create(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| messages || list[dict] || Yes || Chat messages with role/content
|-
| model || str || Yes || Model name (must match served model)
|-
| temperature || float || No || Sampling temperature (0-2)
|-
| max_tokens || int || No || Maximum tokens to generate
|-
| stream || bool || No || Enable streaming (default: False)
|-
| stop || str/list || No || Stop sequences
|-
| top_p || float || No || Nucleus sampling threshold
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| ChatCompletion || object || Full response with choices
|-
| choices[0].message.content || str || Generated text
|-
| usage || dict || Token counts (prompt, completion, total)
|}

== Usage Examples ==

=== Basic Chat Completion ===
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
# Output: "The capital of France is Paris."
</syntaxhighlight>

=== With All Common Parameters ===
<syntaxhighlight lang="python">
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Write a creative story."},
    ],
    temperature=0.9,          # High creativity
    top_p=0.95,               # Nucleus sampling
    max_tokens=500,           # Longer output
    presence_penalty=0.6,     # Encourage variety
    frequency_penalty=0.3,    # Reduce repetition
    stop=["\n\n", "THE END"], # Stop sequences
    n=1,                      # Number of completions
)
</syntaxhighlight>

=== Accessing Full Response ===
<syntaxhighlight lang="python">
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Response structure
print(f"ID: {response.id}")
print(f"Model: {response.model}")
print(f"Created: {response.created}")
print(f"Content: {response.choices[0].message.content}")
print(f"Finish reason: {response.choices[0].finish_reason}")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
</syntaxhighlight>

=== Multiple Completions (n > 1) ===
<syntaxhighlight lang="python">
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Give me a creative name for a pet."}],
    temperature=0.9,
    n=3,  # Generate 3 different responses
)

for i, choice in enumerate(response.choices):
    print(f"Option {i+1}: {choice.message.content}")
</syntaxhighlight>

=== JSON Mode / Structured Output ===
<syntaxhighlight lang="python">
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "Respond in JSON format."},
        {"role": "user", "content": "List 3 colors with hex codes."},
    ],
    response_format={"type": "json_object"},
)

import json
data = json.loads(response.choices[0].message.content)
print(data)
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from openai import OpenAI, APIError, APIConnectionError

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
    )
except APIConnectionError as e:
    print(f"Connection failed: {e}")
except APIError as e:
    print(f"API error: {e.status_code} - {e.message}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_API_Request_Processing]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
