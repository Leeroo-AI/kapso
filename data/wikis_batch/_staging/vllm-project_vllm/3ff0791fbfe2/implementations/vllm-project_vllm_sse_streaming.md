# SSE Streaming (Pattern)

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Streaming|https://platform.openai.com/docs/api-reference/streaming]]
* [[source::Doc|OpenAI Python SDK|https://github.com/openai/openai-python]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Serving]], [[domain::Streaming]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Pattern for consuming streaming responses from vLLM's chat completions API using Server-Sent Events iteration.

=== Description ===

This is a **Pattern Doc** documenting how to consume streaming responses from vLLM. When `stream=True`, the API returns an iterator of chunks containing incremental content.

Each chunk contains a `delta` field with new tokens, allowing real-time display of generation progress.

=== Usage ===

Use streaming when:
* Displaying tokens as they're generated
* Building responsive chat interfaces
* Processing long outputs without waiting
* Implementing early stopping

== Interface Specification ==

=== Required Signature ===
<syntaxhighlight lang="python">
# Enable streaming in request
stream = client.chat.completions.create(
    model="...",
    messages=[...],
    stream=True,  # <-- Enable streaming
)

# Iterate over chunks
for chunk in stream:
    # Each chunk has structure:
    chunk.id           # str: Request ID
    chunk.created      # int: Timestamp
    chunk.model        # str: Model name
    chunk.choices      # list: Completion choices

    # Delta content (may be None)
    delta = chunk.choices[0].delta
    content = delta.content    # str | None: New token(s)
    role = delta.role          # str | None: Role (first chunk only)

    # Finish reason (final chunk only)
    finish_reason = chunk.choices[0].finish_reason  # str | None
</syntaxhighlight>

=== Constraints ===
* First chunk may contain `role` in delta, content is often empty
* Middle chunks contain `content` delta with new tokens
* Final chunk has `finish_reason` set and empty delta
* `[DONE]` signal terminates the stream

== Usage Examples ==

=== Basic Streaming ===
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

print()  # Final newline
</syntaxhighlight>

=== Collecting Full Response ===
<syntaxhighlight lang="python">
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain machine learning."}],
    stream=True,
)

collected_content = []
finish_reason = None

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        collected_content.append(delta.content)
        print(delta.content, end="", flush=True)

    if chunk.choices[0].finish_reason:
        finish_reason = chunk.choices[0].finish_reason

full_response = "".join(collected_content)
print(f"\n\nFinish reason: {finish_reason}")
print(f"Total length: {len(full_response)} chars")
</syntaxhighlight>

=== Async Streaming ===
<syntaxhighlight lang="python">
from openai import AsyncOpenAI
import asyncio

async def stream_chat():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    stream = await client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(stream_chat())
</syntaxhighlight>

=== With Typing Indicator ===
<syntaxhighlight lang="python">
import sys
import time

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Write a story."}],
    stream=True,
)

print("Assistant: ", end="", flush=True)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        # Simulate typing effect
        for char in content:
            print(char, end="", flush=True)
            time.sleep(0.01)  # Small delay between chars

print()
</syntaxhighlight>

=== Early Termination ===
<syntaxhighlight lang="python">
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Count from 1 to 1000."}],
    stream=True,
    max_tokens=1000,
)

collected = []
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        collected.append(content)
        print(content, end="", flush=True)

        # Stop early if we see "50"
        if "50" in "".join(collected):
            print("\n\n[Stopped early]")
            break  # Close connection, stop generation
</syntaxhighlight>

=== Web Application (FastAPI) ===
<syntaxhighlight lang="python">
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

@app.get("/chat")
async def chat(message: str):
    def generate():
        stream = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": message}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")
</syntaxhighlight>

=== Handling Chunk Structure ===
<syntaxhighlight lang="python">
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for i, chunk in enumerate(stream):
    choice = chunk.choices[0]
    delta = choice.delta

    print(f"Chunk {i}:")
    print(f"  id: {chunk.id}")
    print(f"  role: {delta.role}")  # Usually only in first chunk
    print(f"  content: {repr(delta.content)}")
    print(f"  finish_reason: {choice.finish_reason}")
    print()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Streaming_Response]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
