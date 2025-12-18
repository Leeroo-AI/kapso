{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Pooling API]], [[domain::Embeddings]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
HTTP client demonstrating both completion-style and chat-style input formats for vLLM's pooling API endpoint.

=== Description ===
This example shows the dual input format support of vLLM's /pooling endpoint. It demonstrates that the same pooling endpoint can accept inputs in either the simple completion format (with an "input" field containing text) or the structured chat format (with "messages" array following OpenAI's chat API structure). This flexibility allows the pooling API to work seamlessly with different model types and use cases, from simple embeddings to reward models that expect conversational context.

=== Usage ===
Use this example when working with pooling models that support multiple input formats, when migrating from chat-based to completion-based APIs (or vice versa), or when you need to understand how vLLM's pooling endpoint handles different input structures. It's particularly useful for reward models and embedding models that may benefit from structured conversational input.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/pooling/openai_pooling_client.py examples/pooling/pooling/openai_pooling_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server with pooling model
vllm serve internlm/internlm2-1_8b-reward --trust-remote-code

# Run client demonstrating both formats
python openai_pooling_client.py

# With custom configuration
python openai_pooling_client.py \
  --host localhost \
  --port 8000 \
  --model internlm/internlm2-1_8b-reward
</syntaxhighlight>

== Key Concepts ==

=== Completion-Style Input ===
The simple format uses an "input" field with text: {"model": "...", "input": "text"}. This matches the standard embeddings API format.

=== Chat-Style Input ===
The structured format uses "messages" array: {"model": "...", "messages": [{"role": "user", "content": [{"type": "text", "text": "..."}]}]}. This follows OpenAI's chat completions format.

=== Format Flexibility ===
The same /pooling endpoint handles both formats, allowing models to be used in different contexts without changing the server configuration.

=== Reward Model Usage ===
Reward models like internlm2-1_8b-reward can benefit from chat-style input to provide conversational context for reward scoring.

== Usage Examples ==

<syntaxhighlight lang="python">
import requests
import pprint

api_url = "http://localhost:8000/pooling"
model_name = "internlm/internlm2-1_8b-reward"
headers = {"User-Agent": "Test Client"}

# Example 1: Completion-style input
prompt_completion = {
    "model": model_name,
    "input": "vLLM is great!"
}

response = requests.post(api_url, headers=headers, json=prompt_completion)
print("-" * 50)
print("Completion-style Pooling Response:")
pprint.pprint(response.json())
print("-" * 50)

# Example 2: Chat-style input
prompt_chat = {
    "model": model_name,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "vLLM is great!"}
            ]
        }
    ]
}

response = requests.post(api_url, headers=headers, json=prompt_chat)
print("Chat-style Pooling Response:")
pprint.pprint(response.json())
print("-" * 50)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
