{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Online Serving]], [[domain::API Client]], [[domain::Legacy]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates a legacy Python client for vLLM's demonstration API server, supporting both streaming and non-streaming text generation requests.

=== Description ===
This example provides a Python client for interacting with <code>vllm.entrypoints.api_server</code>, a simple HTTP API server included with vLLM for demonstration and basic benchmarking purposes. The client shows how to make both synchronous and streaming generation requests using the <code>/generate</code> endpoint.

'''Important Note:''' This API server is deprecated and intended only for demos and simple tests. For production deployments, use <code>vllm serve</code> with the OpenAI-compatible API instead.

The client demonstrates HTTP request handling, streaming response parsing, and beam search output processing, providing a reference implementation for custom clients if needed.

=== Usage ===
Use this example when:
* Learning about vLLM's legacy API for educational purposes
* Running simple benchmarks or tests with the demo server
* Understanding streaming HTTP response patterns
* Migrating from the legacy API to the OpenAI-compatible API
* Building custom clients for specialized use cases

'''Do not use for production''' - prefer the OpenAI-compatible API with official clients.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/api_client.py examples/online_serving/api_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start the demo API server first
python -m vllm.entrypoints.api_server --model facebook/opt-125m

# In another terminal, run the client
python examples/online_serving/api_client.py \
    --prompt "San Francisco is a" \
    --host localhost \
    --port 8000

# With streaming enabled
python examples/online_serving/api_client.py \
    --prompt "The future of AI is" \
    --stream

# Request multiple beam candidates
python examples/online_serving/api_client.py \
    --prompt "Hello, my name is" \
    --n 3
</syntaxhighlight>

== Key Concepts ==

=== Legacy API Server ===
The <code>vllm.entrypoints.api_server</code> provides:
* Simple HTTP interface on <code>/generate</code> endpoint
* Basic request/response JSON format
* Streaming support via newline-delimited JSON
* Limited configuration options compared to OpenAI API
* No authentication or advanced features

'''Deprecation:''' This server is being phased out in favor of the OpenAI-compatible API.

=== Request Format ===
The client sends JSON requests with:
<syntaxhighlight lang="python">
{
    "prompt": "Text prompt",
    "n": 1,  # Number of beam candidates
    "temperature": 0.0,
    "max_tokens": 16,
    "stream": false  # or true for streaming
}
</syntaxhighlight>

=== Response Formats ===
'''Non-streaming response:'''
<syntaxhighlight lang="json">
{
    "text": ["Generated text 1", "Generated text 2"]
}
</syntaxhighlight>

'''Streaming response:''' Newline-delimited JSON chunks:
<syntaxhighlight lang="json">
{"text": ["Partial ", "Another "]}
{"text": ["Partial text continues", "Another continues"]}
{"text": ["Final text", "Another final"]}
</syntaxhighlight>

=== Beam Search Support ===
The <code>n</code> parameter enables beam search:
* Returns multiple candidate completions
* Each candidate in the "text" array
* Useful for comparing different generation paths

== Usage Examples ==

=== Basic Non-Streaming Request ===
<syntaxhighlight lang="python">
import requests
import json

def post_http_request(prompt: str, api_url: str, n: int = 1, stream: bool = False):
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=stream)
    return response

# Make request
api_url = "http://localhost:8000/generate"
response = post_http_request("Hello, world", api_url)

# Parse response
data = json.loads(response.content)
output = data["text"]
print(f"Generated: {output[0]}")
</syntaxhighlight>

=== Streaming Response Processing ===
<syntaxhighlight lang="python">
def get_streaming_response(response):
    for chunk in response.iter_lines(
        chunk_size=8192,
        decode_unicode=False,
        delimiter=b"\n"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output

# Make streaming request
response = post_http_request(
    "The capital of France is",
    "http://localhost:8000/generate",
    stream=True
)

# Process stream
for text_candidates in get_streaming_response(response):
    print(f"\rGenerated: {text_candidates[0]}", end="", flush=True)
print()  # Final newline
</syntaxhighlight>

=== Multiple Beam Candidates ===
<syntaxhighlight lang="python">
# Request 3 beam candidates
response = post_http_request(
    "Once upon a time",
    "http://localhost:8000/generate",
    n=3
)

data = json.loads(response.content)
candidates = data["text"]

print("Beam candidates:")
for i, candidate in enumerate(candidates):
    print(f"  {i+1}. {candidate}")
</syntaxhighlight>

=== Interactive Streaming Display ===
<syntaxhighlight lang="python">
def clear_line(n: int = 1) -> None:
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)

# Stream with dynamic display update
response = post_http_request(prompt, api_url, n=3, stream=True)

num_printed_lines = 0
for text_candidates in get_streaming_response(response):
    clear_line(num_printed_lines)
    num_printed_lines = 0

    for i, text in enumerate(text_candidates):
        num_printed_lines += 1
        print(f"Beam candidate {i}: {text!r}", flush=True)
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
import requests

try:
    response = post_http_request(prompt, api_url)
    response.raise_for_status()  # Raise exception for HTTP errors

    data = json.loads(response.content)
    if "text" not in data:
        print("Error: Unexpected response format")
    else:
        print(f"Generated: {data['text'][0]}")

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to API server")
    print("Make sure the server is running with:")
    print("  python -m vllm.entrypoints.api_server --model <model_name>")

except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")

except json.JSONDecodeError:
    print("Error: Invalid JSON response")
</syntaxhighlight>

== Migration to OpenAI API ==

=== Equivalent OpenAI Client Code ===
Instead of this legacy API client, use the OpenAI client:

<syntaxhighlight lang="python">
from openai import OpenAI

# Start server with: vllm serve facebook/opt-125m

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# Non-streaming
completion = client.completions.create(
    model="facebook/opt-125m",
    prompt="San Francisco is a",
    max_tokens=16,
    temperature=0.0,
    n=1,
)
print(completion.choices[0].text)

# Streaming
stream = client.completions.create(
    model="facebook/opt-125m",
    prompt="The future of AI is",
    max_tokens=16,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)
</syntaxhighlight>

=== Benefits of OpenAI API ===
* '''Official client libraries''' for multiple languages
* '''Standardized format''' compatible with OpenAI tools
* '''More features''': Function calling, embeddings, chat completions
* '''Better error handling''' and retry logic
* '''Production-ready''' with authentication and monitoring support
* '''Active development''' and long-term support

== Server Configuration ==

=== Starting the Legacy Server ===
<syntaxhighlight lang="bash">
# Basic server
python -m vllm.entrypoints.api_server \
    --model facebook/opt-125m \
    --host 0.0.0.0 \
    --port 8000

# With tensor parallelism
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
</syntaxhighlight>

=== Testing Server Availability ===
<syntaxhighlight lang="bash">
# Check if server is running
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 5}'
</syntaxhighlight>

== Limitations ==

=== Legacy API Limitations ===
* '''No authentication''': No API key validation
* '''Limited features''': No chat format, embeddings, or function calling
* '''No batching control''': Server decides request batching
* '''Minimal error messages''': Less detailed error reporting
* '''No streaming validation''': Client must handle malformed chunks
* '''Deprecated''': Will be removed in future versions

=== Client Limitations ===
* '''No retry logic''': Network failures are not handled
* '''Fixed parameters''': Hard-coded temperature and max_tokens in example
* '''No async support''': Synchronous requests only
* '''Basic parsing''': Minimal validation of responses

== When to Use ==

=== Appropriate Use Cases ===
* Quick prototyping and experimentation
* Simple benchmarking scripts
* Educational examples for understanding HTTP APIs
* Reference implementation for custom clients

=== Better Alternatives ===
* '''Production serving''': Use <code>vllm serve</code> with OpenAI client
* '''Async workloads''': Use OpenAI's AsyncOpenAI client
* '''Chat interfaces''': Use chat completions API
* '''Complex workflows''': Use LangChain or other LLM frameworks with OpenAI integration

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Implementation:vllm-project_vllm_GradioWebserver]]
