{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Serving]], [[domain::Web_APIs]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

External tool documentation for launching the vLLM OpenAI-compatible HTTP server process with model loading and endpoint exposure.

=== Description ===

The `vllm serve` command starts a FastAPI-based HTTP server that exposes OpenAI-compatible endpoints. The server:
- Loads the specified model into GPU memory
- Initializes the AsyncLLMEngine for concurrent request handling
- Exposes REST API endpoints at `/v1/completions`, `/v1/chat/completions`, `/v1/models`
- Handles request queuing, batching, and streaming responses
- Provides health check and metrics endpoints

=== Usage ===

Launch the server when:
- Deploying vLLM as a drop-in OpenAI replacement
- Building applications that consume LLM APIs
- Setting up multi-model serving infrastructure
- Creating development environments for API testing

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/openai/api_server.py
* '''Lines:''' L1-500

=== CLI Interface ===
<syntaxhighlight lang="bash">
# Primary launch command
vllm serve <model_name_or_path>

# Alternative: direct uvicorn launch (advanced)
python -m vllm.entrypoints.openai.api_server --model <model>
</syntaxhighlight>

=== Exposed Endpoints ===
<syntaxhighlight lang="text">
POST /v1/completions          # Text completions (legacy)
POST /v1/chat/completions     # Chat completions (recommended)
GET  /v1/models               # List available models
GET  /health                  # Health check
GET  /version                 # Server version info
GET  /metrics                 # Prometheus metrics (if enabled)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || Model name/path passed to vllm serve
|-
| Configuration || CLI args/env || No || Server and model configuration
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| HTTP Server || Process || uvicorn server listening on configured host:port
|-
| /v1/chat/completions || Endpoint || OpenAI-compatible chat API
|-
| /v1/completions || Endpoint || OpenAI-compatible completion API
|-
| /v1/models || Endpoint || Model listing endpoint
|}

== Usage Examples ==

=== Basic Server Launch ===
<syntaxhighlight lang="bash">
# Launch server (blocks until killed)
vllm serve meta-llama/Llama-3.2-1B-Instruct

# Output:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
</syntaxhighlight>

=== Verify Server Running ===
<syntaxhighlight lang="bash">
# Health check
curl http://localhost:8000/health
# Returns: {"status": "ok"}

# List models
curl http://localhost:8000/v1/models
# Returns: {"data": [{"id": "meta-llama/Llama-3.2-1B-Instruct", ...}]}
</syntaxhighlight>

=== Background Launch with Logging ===
<syntaxhighlight lang="bash">
# Launch in background with log file
nohup vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --port 8000 \
    > vllm.log 2>&1 &

# Check logs
tail -f vllm.log
</syntaxhighlight>

=== Docker Deployment ===
<syntaxhighlight lang="bash">
# Using official vLLM Docker image
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.2-1B-Instruct
</syntaxhighlight>

=== Systemd Service (Production) ===
<syntaxhighlight lang="ini">
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM OpenAI-compatible Server
After=network.target

[Service]
Type=simple
User=vllm
Environment="HF_TOKEN=hf_..."
Environment="CUDA_VISIBLE_DEVICES=0,1"
ExecStart=/usr/local/bin/vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --api-key ${VLLM_API_KEY}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
</syntaxhighlight>

<syntaxhighlight lang="bash">
# Enable and start service
sudo systemctl enable vllm
sudo systemctl start vllm
sudo systemctl status vllm
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Server_Launch]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Server_Environment]]
