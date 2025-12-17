# Server Startup

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

Principle for initializing and starting the vLLM API server, including model loading, engine warmup, and endpoint registration.

=== Description ===

Server Startup encompasses the initialization sequence when launching a vLLM API server:

1. **Argument parsing**: Load CLI arguments and environment variables
2. **Model loading**: Initialize engine with weights and tokenizer
3. **Memory allocation**: Pre-allocate KV cache blocks
4. **Endpoint registration**: Set up FastAPI routes
5. **Server binding**: Start uvicorn on configured host:port

The startup process ensures the model is fully loaded and warmed up before accepting requests.

=== Usage ===

Understand server startup when:
* Troubleshooting slow server initialization
* Monitoring deployment health checks
* Configuring container orchestration (K8s readiness probes)
* Implementing custom server wrappers

== Practical Guide ==

'''Startup Sequence:'''
<syntaxhighlight lang="python">
# Abstract startup flow
def start_vllm_server(args):
    # 1. Parse and validate configuration
    engine_args = EngineArgs.from_cli_args(args)

    # 2. Initialize async engine (loads model)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 3. Create FastAPI application
    app = create_openai_api_app(engine)

    # 4. Start uvicorn server
    uvicorn.run(app, host=args.host, port=args.port)
</syntaxhighlight>

'''Health Check Strategy:'''
* `/health` endpoint returns 200 when ready
* Use for K8s readiness/liveness probes
* Model loading can take minutes for large models

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_vllm_serve_startup]]
