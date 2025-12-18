{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Doc|FastAPI Documentation|https://fastapi.tiangolo.com]]
|-
! Domains
| [[domain::NLP]], [[domain::Serving]], [[domain::Web_APIs]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of starting an HTTP server process that loads the model, initializes the inference engine, and exposes API endpoints for client requests.

=== Description ===

Server Launch is the initialization sequence that transitions from configuration to a running API service. The process involves:

1. **Configuration Parsing:** Reading CLI args, env vars, and config files
2. **Model Loading:** Downloading and loading model weights to GPU
3. **Engine Initialization:** Creating the AsyncLLMEngine for request handling
4. **Endpoint Registration:** Setting up FastAPI routes for API endpoints
5. **Server Start:** Launching uvicorn to accept HTTP connections
6. **Ready Signal:** Completing startup and accepting requests

This process takes 30 seconds to several minutes depending on model size and network speed.

=== Usage ===

Understand server launch when:
- Debugging startup failures or timeouts
- Optimizing cold start times
- Implementing health checks and readiness probes
- Setting up orchestration with Kubernetes
- Managing server lifecycle in production

== Theoretical Basis ==

'''Launch Sequence:'''

<syntaxhighlight lang="python">
# Conceptual server launch flow
async def launch_server(config):
    # 1. Parse configuration
    engine_args = EngineArgs.from_cli(config)

    # 2. Download model (if needed)
    if not is_model_cached(engine_args.model):
        download_model(engine_args.model)  # Can take minutes

    # 3. Initialize async engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # - Loads model weights to GPU
    # - Allocates KV cache
    # - Initializes tokenizer

    # 4. Create FastAPI app
    app = FastAPI()

    # 5. Register endpoints
    @app.post("/v1/chat/completions")
    async def chat_completion(request):
        return await engine.generate(request)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # 6. Start uvicorn server
    uvicorn.run(app, host=config.host, port=config.port)
</syntaxhighlight>

'''Startup Phases:'''

| Phase | Description | Duration |
|-------|-------------|----------|
| Config parse | Read CLI/env/file config | < 1s |
| Model download | Download from HuggingFace | 0-30+ min |
| Weight loading | Load weights to GPU | 10s-5min |
| KV cache alloc | Allocate GPU memory for cache | 1-10s |
| Tokenizer init | Load tokenizer | 1-5s |
| Server bind | Bind to host:port | < 1s |

'''Health Check Design:'''

<syntaxhighlight lang="python">
# Kubernetes-style probes
# Liveness: Is the process alive?
GET /health -> 200 OK

# Readiness: Can it serve requests?
GET /v1/models -> 200 OK (with model list)

# Startup: Has initialization completed?
# (typically same as readiness for vLLM)
</syntaxhighlight>

'''Graceful Shutdown:'''

<syntaxhighlight lang="python">
# Server handles SIGTERM/SIGINT
def shutdown():
    # 1. Stop accepting new requests
    server.shutdown()

    # 2. Wait for in-flight requests (with timeout)
    await engine.wait_for_pending(timeout=30)

    # 3. Release resources
    engine.cleanup()
    torch.cuda.empty_cache()
</syntaxhighlight>

'''Cold Start Optimization:'''

<syntaxhighlight lang="python">
# Strategies to reduce cold start time:

# 1. Pre-download models in image build
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('meta-llama/Llama-3.2-1B-Instruct')"

# 2. Use quantized models (smaller weights)
vllm serve TheBloke/Llama-2-7B-AWQ --quantization awq

# 3. Use tensor parallel for large models
vllm serve meta-llama/Llama-2-70B --tensor-parallel-size 4
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_api_server_run]]
