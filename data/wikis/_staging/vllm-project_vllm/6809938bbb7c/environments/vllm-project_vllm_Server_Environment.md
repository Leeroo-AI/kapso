# Server Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Server Deployment|https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::API_Server]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-01-15 14:00 GMT]]
|}

== Overview ==

Linux server environment with uvicorn, FastAPI, and GPU support for deploying vLLM as an OpenAI-compatible HTTP API server.

=== Description ===

This environment provides the runtime context for deploying vLLM as a production HTTP server. It combines the GPU environment requirements with web server components (uvicorn, FastAPI) and optional authentication/security features. The server exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and supports features like streaming, tool calling, and multimodal inputs.

=== Usage ===

Use this environment when deploying vLLM as an **HTTP API server**:
- Running `vllm serve <model>` CLI command
- Programmatic server launch via `uvicorn`
- Production deployments with Docker/Kubernetes
- Multi-instance deployments with load balancing

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04/22.04) || Production deployments require Linux
|-
| Hardware || GPU requirements same as GPU_Environment || See [[Environment:vllm-project_vllm_GPU_Environment]]
|-
| Network || Port 8000 (default) available || Configurable via `--port`
|-
| Memory || 32GB+ system RAM || Server overhead + model loading
|-
| Disk || 100GB+ SSD || Model weights + logging
|}

== Dependencies ==

=== System Packages ===
* All packages from [[Environment:vllm-project_vllm_GPU_Environment]]

=== Python Packages ===
* `uvicorn` >= 0.22.0 (ASGI server)
* `fastapi` >= 0.100.0 (web framework)
* `uvloop` (optional, for better async performance on Linux)
* `httptools` (optional, for faster HTTP parsing)
* `prometheus-client` (optional, for metrics)
* `opentelemetry-api` (optional, for tracing)

== Credentials ==

The following environment variables configure server security:
* `VLLM_API_KEY`: Optional API key for authentication (set to enable `--api-key` validation)
* `HF_TOKEN`: HuggingFace token for gated model access

== Quick Install ==

<syntaxhighlight lang="bash">
# Install vLLM with server dependencies
pip install vllm[server]

# Or install server dependencies separately
pip install uvicorn fastapi uvloop httptools

# Launch server
vllm serve meta-llama/Llama-3.2-1B --port 8000

# With API key authentication
VLLM_API_KEY=your-secret-key vllm serve meta-llama/Llama-3.2-1B --api-key your-secret-key
</syntaxhighlight>

== Code Evidence ==

Server launch configuration from `vllm/entrypoints/openai/cli_args.py`:
<syntaxhighlight lang="python">
parser.add_argument("--host", type=str, default="0.0.0.0",
                    help="Host to bind the server to")
parser.add_argument("--port", type=int, default=8000,
                    help="Port to bind the server to")
parser.add_argument("--api-key", type=str, default=None,
                    help="API key for server authentication")
</syntaxhighlight>

HTTP timeout configuration from `vllm/envs.py:841-843`:
<syntaxhighlight lang="python">
# Timeout in seconds for keeping HTTP connections alive in API server
"VLLM_HTTP_TIMEOUT_KEEP_ALIVE": lambda: int(
    os.environ.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5")
),
</syntaxhighlight>

Server keep-alive behavior from `vllm/envs.py:817-819`:
<syntaxhighlight lang="python">
# If set, the OpenAI API server will stay alive even after the underlying
# AsyncLLMEngine errors and stops serving requests
"VLLM_KEEP_ALIVE_ON_ENGINE_DEATH": lambda: bool(
    int(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", "0"))
),
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Address already in use (port 8000)` || Another process using the port || Use `--port` to specify different port, or kill existing process
|-
|| `uvicorn: command not found` || uvicorn not installed || `pip install uvicorn`
|-
|| `CUDA out of memory during server startup` || Insufficient VRAM || Reduce `--gpu-memory-utilization` or use quantization
|-
|| `401 Unauthorized` || API key mismatch || Ensure client uses same key as `--api-key`
|-
|| `Connection reset by peer` || Server crashed or timeout || Check server logs, increase `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`
|}

== Compatibility Notes ==

* '''uvloop:''' Recommended for Linux deployments; not available on Windows.
* '''Multi-GPU:''' Use `--tensor-parallel-size N` for tensor parallelism across N GPUs.
* '''Docker:''' Use `--host 0.0.0.0` to expose server outside container.
* '''Kubernetes:''' Set resource limits and use readiness/liveness probes on `/health` endpoint.
* '''Load Balancing:''' Multiple vLLM instances can be deployed behind a load balancer; each instance is stateless.

== Environment Variables ==

Key server-specific environment variables:
* `VLLM_HOST_IP`: IP address for distributed deployment coordination
* `VLLM_PORT`: Default port (overridden by `--port`)
* `VLLM_API_KEY`: API authentication key
* `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`: HTTP keep-alive timeout (default: 5 seconds)
* `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH`: Keep server running if engine fails (default: false)
* `VLLM_SERVER_DEV_MODE`: Enable development endpoints like `/reset_prefix_cache`

== Related Pages ==

* [[requires_env::Implementation:vllm-project_vllm_vllm_serve_args]]
* [[requires_env::Implementation:vllm-project_vllm_api_server_run]]
