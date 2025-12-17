{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference]]
|-
! Domains
| [[domain::LLMs]], [[domain::Serving]], [[domain::API]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

End-to-end process for deploying vLLM as an OpenAI-compatible API server for production inference serving with chat completions, function calling, and streaming support.

=== Description ===

This workflow covers deploying vLLM as a REST API server that provides OpenAI-compatible endpoints. The server supports chat completions, text completions, embeddings, and advanced features like function calling and reasoning model outputs. It is designed for production deployments where multiple clients need concurrent access to LLM inference.

Key capabilities:
* **OpenAI compatibility**: Drop-in replacement for OpenAI API
* **Streaming**: Server-sent events for real-time token streaming
* **Function calling**: Tool use and structured outputs
* **Multi-model**: Serve multiple models or LoRA adapters
* **Metrics**: Prometheus metrics for monitoring

=== Usage ===

Execute this workflow when you need to:
* Deploy an LLM inference service for multiple clients
* Integrate LLM capabilities into existing applications using OpenAI SDK
* Build chatbots, assistants, or RAG pipelines
* Serve models with function calling or tool use capabilities

Ideal for production deployments where you need a scalable, monitored API endpoint.

== Execution Steps ==

=== Step 1: Configure Server Arguments ===
[[step::Principle:vllm-project_vllm_Server_Configuration]]

Configure the vLLM server with model selection, serving parameters, and API settings. This includes specifying the model, host/port binding, API key authentication, and engine configuration.

'''Key considerations:'''
* Specify model path or HuggingFace ID
* Set `--host` and `--port` for network binding
* Configure `--api-key` for authentication
* Enable `--enable-auto-tool-choice` for function calling
* Set parallelism with `--tensor-parallel-size`

=== Step 2: Start API Server ===
[[step::Principle:vllm-project_vllm_Server_Startup]]

Launch the vLLM API server using the `vllm serve` command or Python API. The server initializes the model, starts the async engine, and begins accepting HTTP requests on the configured endpoint.

'''Startup process:'''
1. Model and tokenizer are loaded into GPU memory
2. KV cache is pre-allocated based on configuration
3. FastAPI server starts with OpenAI-compatible routes
4. Health check endpoint becomes available at `/health`
5. Model list endpoint at `/v1/models`

=== Step 3: Configure API Client ===
[[step::Principle:vllm-project_vllm_API_Client_Setup]]

Set up the client to connect to the vLLM server. This typically uses the OpenAI Python SDK with a custom `base_url` pointing to the vLLM server. The client handles request formatting, authentication, and response parsing.

'''Client configuration:'''
* Set `base_url` to vLLM server endpoint (e.g., `http://localhost:8000/v1`)
* Configure `api_key` if server authentication is enabled
* Standard OpenAI SDK methods work without modification

=== Step 4: Format Chat Messages ===
[[step::Principle:vllm-project_vllm_Chat_Formatting]]

Structure conversation history as chat messages with roles. The server applies the model's chat template to convert messages into the appropriate prompt format for generation.

'''Message structure:'''
* `system`: System prompt defining assistant behavior
* `user`: User messages and queries
* `assistant`: Previous assistant responses
* `tool`: Tool/function call results

=== Step 5: Execute Inference Request ===
[[step::Principle:vllm-project_vllm_API_Request_Processing]]

Send inference requests to the server via HTTP POST. The server queues requests, processes them using continuous batching, and returns responses. Supports both synchronous and streaming modes.

'''Request flow:'''
1. Client sends POST to `/v1/chat/completions` or `/v1/completions`
2. Server validates and queues the request
3. Continuous batching processes requests efficiently
4. Response returned as JSON or SSE stream

=== Step 6: Handle Streaming Response ===
[[step::Principle:vllm-project_vllm_Streaming_Response]]

Process streaming responses using server-sent events (SSE). Each chunk contains a delta with newly generated tokens, allowing real-time display of generation progress.

'''Streaming handling:'''
* Enable with `stream=True` in request
* Iterate over response chunks
* Each chunk contains `choices[0].delta.content`
* Final chunk includes `finish_reason`

== Execution Diagram ==
{{#mermaid:graph TD
    A[Configure Server Arguments] --> B[Start API Server]
    B --> C[Configure API Client]
    C --> D[Format Chat Messages]
    D --> E[Execute Inference Request]
    E --> F[Handle Streaming Response]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_Server_Configuration]]
* [[step::Principle:vllm-project_vllm_Server_Startup]]
* [[step::Principle:vllm-project_vllm_API_Client_Setup]]
* [[step::Principle:vllm-project_vllm_Chat_Formatting]]
* [[step::Principle:vllm-project_vllm_API_Request_Processing]]
* [[step::Principle:vllm-project_vllm_Streaming_Response]]
