# vllm serve

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Serving]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Concrete tool for configuring and launching the vLLM OpenAI-compatible API server with model, network, and performance settings.

=== Description ===

`vllm serve` is the CLI command for starting an OpenAI-compatible API server. It accepts all engine configuration arguments plus server-specific options for networking, authentication, and API behavior.

The server uses FastAPI with uvicorn for async HTTP handling and integrates with the AsyncLLMEngine for continuous batching.

=== Usage ===

Use `vllm serve` to:
* Deploy models for production inference
* Create OpenAI API-compatible endpoints
* Serve models with streaming support
* Handle concurrent requests with automatic batching

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm]
* '''File:''' vllm/entrypoints/openai/api_server.py

=== Signature ===
<syntaxhighlight lang="bash">
vllm serve <model> [OPTIONS]

# Key options:
#   --host TEXT                Host to bind to (default: 0.0.0.0)
#   --port INT                 Port to bind to (default: 8000)
#   --api-key TEXT             API key for authentication
#   --served-model-name TEXT   Model name returned by API
#   --tensor-parallel-size INT Number of GPUs for TP
#   --gpu-memory-utilization FLOAT  GPU memory fraction (default: 0.9)
#   --max-model-len INT        Maximum context length
#   --dtype TEXT               Model dtype (auto/float16/bfloat16)
#   --quantization TEXT        Quantization method
#   --enable-lora              Enable LoRA adapter support
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# CLI command - no Python import needed
vllm serve meta-llama/Llama-3.1-8B-Instruct
</syntaxhighlight>

== I/O Contract ==

=== Inputs (CLI Arguments) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || HuggingFace model name or path
|-
| --host || str || No || Bind address (default: 0.0.0.0)
|-
| --port || int || No || Port number (default: 8000)
|-
| --api-key || str || No || Required API key for requests
|-
| --tensor-parallel-size || int || No || GPUs for tensor parallelism
|-
| --gpu-memory-utilization || float || No || GPU memory fraction
|}

=== Outputs (API Endpoints) ===
{| class="wikitable"
|-
! Endpoint !! Method !! Description
|-
| /v1/chat/completions || POST || Chat completion (OpenAI-compatible)
|-
| /v1/completions || POST || Text completion (OpenAI-compatible)
|-
| /v1/embeddings || POST || Text embeddings
|-
| /v1/models || GET || List available models
|-
| /health || GET || Server health check
|}

== Usage Examples ==

=== Basic Server Start ===
<syntaxhighlight lang="bash">
# Start server with default settings
vllm serve meta-llama/Llama-3.1-8B-Instruct

# Server will be available at http://localhost:8000
</syntaxhighlight>

=== Custom Host and Port ===
<syntaxhighlight lang="bash">
# Bind to specific interface and port
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8080
</syntaxhighlight>

=== With API Key Authentication ===
<syntaxhighlight lang="bash">
# Require API key for all requests
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --api-key "sk-your-secret-key"
</syntaxhighlight>

=== Multi-GPU Serving ===
<syntaxhighlight lang="bash">
# Serve 70B model across 4 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
</syntaxhighlight>

=== With LoRA Support ===
<syntaxhighlight lang="bash">
# Enable runtime LoRA adapter loading
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules my-lora=/path/to/lora
</syntaxhighlight>

=== Production Configuration ===
<syntaxhighlight lang="bash">
# Full production setup
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "$VLLM_API_KEY" \
    --served-model-name "llama-3.1-8b" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --disable-log-requests
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Server_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
