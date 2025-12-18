# Environment Index: vllm-project_vllm

> Tracks Environment pages and which pages require them.
> **Update IMMEDIATELY** after creating or modifying a Environment page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_GPU_Environment | [→](./environments/vllm-project_vllm_GPU_Environment.md) | ✅Impl:vllm-project_vllm_LLM_init, ✅Impl:vllm-project_vllm_LLM_generate, ✅Impl:vllm-project_vllm_EngineArgs_lora, ✅Impl:vllm-project_vllm_LLMEngine_add_request_lora, ✅Impl:vllm-project_vllm_Scheduler_lora_batching, ✅Impl:vllm-project_vllm_EngineArgs_vlm, ✅Impl:vllm-project_vllm_LLM_generate_mm, ✅Impl:vllm-project_vllm_LLM_speculative, ✅Impl:vllm-project_vllm_LLM_generate_spec, ✅Impl:vllm-project_vllm_LLM_generate_structured | Linux + CUDA 11.8+ or ROCm for GPU-accelerated LLM inference |
| vllm-project_vllm_Python_Environment | [→](./environments/vllm-project_vllm_Python_Environment.md) | ✅Impl:vllm-project_vllm_SamplingParams_init, ✅Impl:vllm-project_vllm_PromptType_usage, ✅Impl:vllm-project_vllm_RequestOutput_usage, ✅Impl:vllm-project_vllm_LoRARequest_init, ✅Impl:vllm-project_vllm_RequestOutput_lora, ✅Impl:vllm-project_vllm_MultiModalData_image, ✅Impl:vllm-project_vllm_VLM_prompt_format, ✅Impl:vllm-project_vllm_RequestOutput_vlm, ✅Impl:vllm-project_vllm_SpeculativeMethod_choice, ✅Impl:vllm-project_vllm_SpeculativeConfig_init, ✅Impl:vllm-project_vllm_get_metrics_spec, ✅Impl:vllm-project_vllm_StructuredOutputsParams_types, ✅Impl:vllm-project_vllm_StructuredOutputsParams_init, ✅Impl:vllm-project_vllm_SamplingParams_structured, ✅Impl:vllm-project_vllm_structured_output_parse | Python 3.9+ with transformers, msgspec, pydantic |
| vllm-project_vllm_Server_Environment | [→](./environments/vllm-project_vllm_Server_Environment.md) | ✅Impl:vllm-project_vllm_vllm_serve_args, ✅Impl:vllm-project_vllm_api_server_run | Linux server with uvicorn, FastAPI for HTTP API deployment |
| vllm-project_vllm_Client_Environment | [→](./environments/vllm-project_vllm_Client_Environment.md) | ✅Impl:vllm-project_vllm_OpenAI_client_init, ✅Impl:vllm-project_vllm_chat_completions_create, ✅Impl:vllm-project_vllm_ChatCompletion_processing | Python with OpenAI client library for API consumption |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
