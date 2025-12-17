# API Client Setup

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

Principle for configuring API clients to communicate with vLLM's OpenAI-compatible server endpoints.

=== Description ===

API Client Setup enables communication between client applications and the vLLM server. Since vLLM exposes an OpenAI-compatible API, the official OpenAI Python SDK can be used with minimal configuration changes.

Key setup components:
* **Base URL**: Point to vLLM server instead of OpenAI
* **API Key**: Use vLLM's configured key (or placeholder)
* **Model name**: Match served model name
* **Timeout settings**: Configure for long-running requests

=== Usage ===

Set up API clients when:
* Building applications that call vLLM servers
* Migrating from OpenAI API to self-hosted vLLM
* Creating test clients for server validation
* Integrating vLLM into existing OpenAI-based workflows

== Theoretical Basis ==

'''OpenAI Compatibility:'''

vLLM implements the OpenAI API specification, enabling:
<syntaxhighlight lang="python">
# Standard OpenAI code
client = OpenAI(api_key="sk-xxx")
response = client.chat.completions.create(...)

# Works with vLLM by changing base_url
client = OpenAI(base_url="http://vllm:8000/v1", api_key="any")
response = client.chat.completions.create(...)  # Same code!
</syntaxhighlight>

'''Supported Endpoints:'''
* `/v1/chat/completions` - Chat format
* `/v1/completions` - Legacy completion format
* `/v1/embeddings` - Text embeddings
* `/v1/models` - Model listing

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_OpenAI_Client]]
