{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|OpenAI Compatible Server|https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html]]
|-
! Domains
| [[domain::LLM_Serving]], [[domain::API_Server]], [[domain::OpenAI_API]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for deploying a vLLM model as an OpenAI-compatible HTTP API server and consuming it from Python clients.

=== Description ===
This workflow covers the production deployment pattern for serving LLMs through vLLM's OpenAI-compatible API server. The server exposes `/v1/chat/completions`, `/v1/completions`, and `/v1/models` endpoints that are compatible with OpenAI's Python client library. This enables drop-in replacement of OpenAI API calls with local inference, supporting streaming responses, tool calling, and multi-turn conversations.

=== Usage ===
Execute this workflow when you need to serve LLM inference as an HTTP service for applications, enable multiple clients to share a single GPU-backed inference engine, or integrate with existing systems that use the OpenAI API format. This is the recommended pattern for production deployments and microservices architectures.

== Execution Steps ==

=== Step 1: Server Configuration ===
[[step::Principle:vllm-project_vllm_Server_Configuration]]

Configure the vLLM server using command-line arguments or environment variables. Key settings include model path, tensor parallelism for multi-GPU, port binding, and API-specific options like chat templates and tool calling configuration.

'''Key configuration options:'''
* `--model` - HuggingFace model name or local path
* `--tensor-parallel-size` - Number of GPUs for model sharding
* `--port` - HTTP server port (default 8000)
* `--chat-template` - Custom Jinja2 template for chat formatting
* `--enable-auto-tool-choice` - Enable tool/function calling support
* `--api-key` - Optional API key for authentication

=== Step 2: Server Launch ===
[[step::Principle:vllm-project_vllm_Server_Launch]]

Start the vLLM server using the `vllm serve` command or programmatically via Python. The server initializes the model, sets up the FastAPI application, and begins listening for HTTP requests. Startup includes model loading, KV cache allocation, and optional CUDA graph compilation.

'''Launch methods:'''
* CLI: `vllm serve <model-name> [options]`
* Python: Using `AsyncLLMEngine` with uvicorn
* Docker: Via official vLLM container images
* Ray Serve: For managed deployment with autoscaling

=== Step 3: Client Initialization ===
[[step::Principle:vllm-project_vllm_OpenAI_Client_Setup]]

Initialize an OpenAI Python client configured to connect to the vLLM server instead of OpenAI's servers. This requires setting the `base_url` to point to your vLLM server and optionally configuring the API key.

'''Client configuration:'''
* Set `base_url` to vLLM server URL (e.g., `http://localhost:8000/v1`)
* API key can be set to any value if server doesn't require authentication
* Use standard `OpenAI` class from the `openai` Python package
* Query `/v1/models` to get available model names

=== Step 4: Chat Completion Request ===
[[step::Principle:vllm-project_vllm_Chat_Completion_API]]

Send chat completion requests using the OpenAI client's `chat.completions.create()` method. Format messages as a list of role/content dictionaries following the OpenAI chat format. The server processes requests using continuous batching for optimal throughput.

'''Request components:'''
* `messages` - List of message dictionaries with `role` and `content`
* `model` - Model identifier (from `/v1/models` endpoint)
* `temperature`, `max_tokens` - Generation parameters
* `stream` - Enable streaming for real-time token output
* `tools` - Function definitions for tool calling

=== Step 5: Response Processing ===
[[step::Principle:vllm-project_vllm_Response_Handling]]

Process the server response, which follows the OpenAI response format. Handle both streaming (Server-Sent Events) and non-streaming responses. Extract generated text, handle tool calls, and manage multi-turn conversation state.

'''Response handling patterns:'''
* Non-streaming: Access `response.choices[0].message.content`
* Streaming: Iterate over response chunks for real-time output
* Tool calls: Check `response.choices[0].message.tool_calls` for function invocations
* Usage tracking: Access `response.usage` for token counts

== Execution Diagram ==
{{#mermaid:graph TD
    A[Server Configuration] --> B[Server Launch]
    B --> C[Client Initialization]
    C --> D[Chat Completion Request]
    D --> E[Response Processing]
    E -->|Multi-turn| D
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_Server_Configuration]]
* [[step::Principle:vllm-project_vllm_Server_Launch]]
* [[step::Principle:vllm-project_vllm_OpenAI_Client_Setup]]
* [[step::Principle:vllm-project_vllm_Chat_Completion_API]]
* [[step::Principle:vllm-project_vllm_Response_Handling]]
