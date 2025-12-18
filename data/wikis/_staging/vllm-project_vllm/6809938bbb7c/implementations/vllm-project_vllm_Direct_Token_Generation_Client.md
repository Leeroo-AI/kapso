{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::API]], [[domain::Token Generation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
HTTP client demonstrating direct token ID generation bypassing text tokenization on the server side.

=== Description ===
This minimal example shows how to send pre-tokenized input (token IDs) directly to vLLM's generation endpoint and receive raw token IDs back without server-side detokenization. It uses the Transformers library to apply chat templates client-side, then sends the resulting token IDs to vLLM's /inference/v1/generate endpoint. This approach is useful for fine-grained control over tokenization, reducing server processing overhead, or implementing custom prompt formatting.

=== Usage ===
Use this example when you need precise control over tokenization, want to avoid double-tokenization overhead, need to inspect or manipulate token IDs before generation, or are implementing custom prompt formats that require client-side template application. It's particularly relevant for advanced use cases where standard text-based APIs don't provide sufficient control.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/token_generation_client.py examples/online_serving/token_generation_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server
vllm serve Qwen/Qwen3-0.6B

# Run the token generation client
python token_generation_client.py
</syntaxhighlight>

== Key Concepts ==

=== Pre-Tokenization ===
The client uses Transformers' AutoTokenizer to apply chat templates and convert messages to token IDs before sending to the server. This moves tokenization responsibility from server to client.

=== Token ID Payload ===
Instead of sending text prompts, the request payload contains a "token_ids" field with the pre-computed token sequence. The server directly generates from these IDs.

=== Detokenize Parameter ===
Setting "detokenize": false in sampling_params tells vLLM to return raw token IDs instead of decoded text. The client then handles decoding.

=== Generation Endpoint ===
Uses the /inference/v1/generate endpoint rather than standard OpenAI-compatible endpoints. This endpoint accepts token_ids and provides more direct control over generation.

== Usage Examples ==

<syntaxhighlight lang="python">
import httpx
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "http://localhost:8000/inference/v1/generate"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How many countries are in the EU?"}
]

# Apply chat template to get token IDs
token_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False
)

# Prepare payload with token IDs
payload = {
    "model": MODEL_NAME,
    "token_ids": token_ids,
    "sampling_params": {
        "max_tokens": 24,
        "temperature": 0.2,
        "detokenize": False  # Return token IDs, not text
    },
    "stream": False
}

# Send request
client = httpx.Client(timeout=600)
resp = client.post(GEN_ENDPOINT, json=payload)
data = resp.json()

# Decode token IDs client-side
generated_token_ids = data["choices"][0]["token_ids"]
result = tokenizer.decode(generated_token_ids)
print(result)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
