# vllm serve (Startup)

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Serving]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Concrete tool for starting the vLLM OpenAI-compatible server process and managing its lifecycle.

=== Description ===

The `vllm serve` command handles the complete server lifecycle:
* Model downloading and loading
* Engine initialization and memory allocation
* FastAPI application setup
* uvicorn server startup
* Signal handling for graceful shutdown

This implementation documents the startup behavior and lifecycle management.

=== Usage ===

The server startup is automatic when running `vllm serve`. Monitor startup through:
* Console output for progress
* `/health` endpoint for readiness
* Log files for debugging

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm]
* '''File:''' vllm/entrypoints/openai/api_server.py

=== Signature ===
<syntaxhighlight lang="bash">
# Primary startup command
vllm serve <model> [options]

# The server outputs startup progress:
# INFO:     Started server process [PID]
# INFO:     Waiting for application startup
# INFO:     Loading model...
# INFO:     Model loaded successfully
# INFO:     Application startup complete
# INFO:     Uvicorn running on http://0.0.0.0:8000
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# For programmatic server startup
from vllm.entrypoints.openai.api_server import run_server
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || Model to load and serve
|-
| --host || str || No || Bind address
|-
| --port || int || No || Bind port
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| HTTP server || Process || Running uvicorn server
|-
| /health || Endpoint || Returns 200 when ready
|-
| Logs || stdout/stderr || Startup progress and errors
|}

== Usage Examples ==

=== Basic Startup ===
<syntaxhighlight lang="bash">
# Start and wait for ready
vllm serve meta-llama/Llama-3.1-8B-Instruct

# In another terminal, check readiness
curl http://localhost:8000/health
# Returns: {"status":"healthy"}
</syntaxhighlight>

=== Docker Startup with Health Check ===
<syntaxhighlight lang="dockerfile">
FROM vllm/vllm-openai:latest

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct"]
</syntaxhighlight>

=== Kubernetes Readiness Probe ===
<syntaxhighlight lang="yaml">
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct"]
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60  # Model loading takes time
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
</syntaxhighlight>

=== Programmatic Startup ===
<syntaxhighlight lang="python">
import asyncio
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs

async def main():
    # Programmatic server startup
    args = AsyncEngineArgs(
        model="meta-llama/Llama-3.1-8B-Instruct",
        host="0.0.0.0",
        port=8000,
    )
    await run_server(args)

if __name__ == "__main__":
    asyncio.run(main())
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Server_Startup]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
