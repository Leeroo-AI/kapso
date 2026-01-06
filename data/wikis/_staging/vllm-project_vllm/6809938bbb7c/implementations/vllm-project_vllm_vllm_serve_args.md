{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Serving]], [[domain::CLI]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

External tool documentation for configuring the vLLM OpenAI-compatible server via command-line arguments and environment variables.

=== Description ===

The `vllm serve` command-line interface provides comprehensive configuration options for deploying vLLM as an OpenAI-compatible API server. Configuration can be provided via:
- Command-line arguments (highest priority)
- Environment variables (VLLM_* prefix)
- Configuration files (YAML format)

Key configuration categories include server networking, model loading, tensor parallelism, memory management, LoRA support, and API authentication.

=== Usage ===

Use `vllm serve` CLI arguments when:
- Deploying vLLM as a production API server
- Configuring multi-GPU tensor parallelism
- Setting up authentication with API keys
- Enabling LoRA adapter serving
- Customizing chat templates for specific models

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/openai/cli_args.py (CLI args), vllm/engine/arg_utils.py (EngineArgs)
* '''Lines:''' L1-200 (cli_args.py), L1-300 (arg_utils.py)

=== CLI Interface ===
<syntaxhighlight lang="bash">
vllm serve <model> [options]

# Core options
--model MODEL              # HuggingFace model name or local path
--host HOST                # Server bind address (default: 0.0.0.0)
--port PORT                # Server port (default: 8000)
--api-key KEY              # API key for authentication

# Model configuration
--tensor-parallel-size N   # Number of GPUs for tensor parallelism
--dtype TYPE               # Model dtype (auto, float16, bfloat16)
--quantization METHOD      # Quantization (awq, gptq, fp8, etc.)
--max-model-len N          # Maximum sequence length

# Memory configuration
--gpu-memory-utilization F # GPU memory fraction (default: 0.9)
--swap-space GB            # CPU swap space in GB

# LoRA configuration
--enable-lora              # Enable LoRA adapter serving
--max-loras N              # Max concurrent LoRA adapters
--max-lora-rank N          # Maximum LoRA rank supported

# Chat configuration
--chat-template PATH       # Custom Jinja2 chat template
--response-role ROLE       # Role name for model responses

# Server configuration
--uvicorn-log-level LEVEL  # Logging level (debug, info, warning, error)
--disable-log-requests     # Disable request logging
</syntaxhighlight>

=== Environment Variables ===
<syntaxhighlight lang="bash">
# Common environment variables
export VLLM_API_KEY="your-secret-key"
export VLLM_HOST="0.0.0.0"
export VLLM_PORT="8000"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_TOKEN="hf_..."  # For gated models
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || HuggingFace model name or local path
|-
| --host || str || No || Bind address (default: 0.0.0.0)
|-
| --port || int || No || Server port (default: 8000)
|-
| --tensor-parallel-size || int || No || Number of GPUs (default: 1)
|-
| --api-key || str || No || API authentication key
|-
| --enable-lora || flag || No || Enable LoRA support
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Server process || Process || Running HTTP server on specified host:port
|-
| API endpoints || HTTP || /v1/completions, /v1/chat/completions, /v1/models, /health
|}

== Usage Examples ==

=== Basic Server Launch ===
<syntaxhighlight lang="bash">
# Start server with minimal configuration
vllm serve meta-llama/Llama-3.2-1B-Instruct

# Server will be available at http://localhost:8000
</syntaxhighlight>

=== Multi-GPU with Authentication ===
<syntaxhighlight lang="bash">
# Deploy on 4 GPUs with API key authentication
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --api-key "sk-vllm-secret-key" \
    --port 8080 \
    --gpu-memory-utilization 0.85
</syntaxhighlight>

=== LoRA-Enabled Server ===
<syntaxhighlight lang="bash">
# Enable LoRA adapter serving
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --lora-extra-vocab-size 256
</syntaxhighlight>

=== Quantized Model Serving ===
<syntaxhighlight lang="bash">
# Serve AWQ quantized model
vllm serve TheBloke/Llama-2-70B-AWQ \
    --quantization awq \
    --tensor-parallel-size 2 \
    --dtype float16
</syntaxhighlight>

=== Custom Chat Template ===
<syntaxhighlight lang="bash">
# Use custom chat template
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
    --chat-template /path/to/template.jinja \
    --response-role assistant
</syntaxhighlight>

=== Production Configuration ===
<syntaxhighlight lang="bash">
# Production deployment with all optimizations
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --api-key "${VLLM_API_KEY}" \
    --uvicorn-log-level warning \
    --disable-log-requests
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Server_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Server_Environment]]
