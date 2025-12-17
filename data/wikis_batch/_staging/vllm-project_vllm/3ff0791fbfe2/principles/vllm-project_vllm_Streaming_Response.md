# Streaming Response

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Streaming|https://platform.openai.com/docs/api-reference/streaming]]
* [[source::Doc|Server-Sent Events|https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Serving]], [[domain::Streaming]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Principle for receiving token-by-token generation output through Server-Sent Events (SSE) for real-time response display.

=== Description ===

Streaming Response enables incremental delivery of generated text as tokens are produced, rather than waiting for complete generation. This provides:

* **Lower perceived latency**: First tokens appear immediately
* **Better UX**: Users see progress in real-time
* **Early termination**: Stop generation if output is unwanted
* **Memory efficiency**: Process tokens incrementally

vLLM implements OpenAI-compatible SSE streaming over HTTP.

=== Usage ===

Enable streaming when:
* Building interactive chat applications
* Displaying real-time generation progress
* Implementing typing indicators
* Processing long outputs incrementally

== Theoretical Basis ==

'''Server-Sent Events Protocol:'''
<syntaxhighlight lang="text">
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" world"}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
</syntaxhighlight>

'''Streaming vs Non-Streaming:'''
* Non-streaming: Single response after all tokens generated
* Streaming: Multiple chunks, each with new token(s)
* Delta format: Only new content in each chunk

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_sse_streaming]]
