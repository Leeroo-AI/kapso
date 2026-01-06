# Workflow Index: vllm-project_vllm

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Rough APIs |
|----------|-------|------------|------------|
| Basic_Offline_Inference | 5 | 5 | LLM, SamplingParams, generate() |
| OpenAI_Compatible_Serving | 5 | 5 | vllm serve, OpenAI client, chat.completions |
| Multi_LoRA_Inference | 5 | 5 | LLMEngine, LoRARequest, enable_lora |
| Vision_Language_Inference | 5 | 5 | LLM, EngineArgs, limit_mm_per_prompt |
| Speculative_Decoding | 5 | 5 | LLM, speculative_config, EAGLE/ngram |
| Structured_Output_Generation | 5 | 5 | StructuredOutputsParams, SamplingParams |

---

## Workflow: vllm-project_vllm_Basic_Offline_Inference

**File:** [→](./workflows/vllm-project_vllm_Basic_Offline_Inference.md)
**Description:** Batch text generation with vLLM's offline LLM class on local GPU hardware.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Model Initialization | LLM_Class_Initialization | `LLM(model=...)` | vllm/__init__.py, vllm/entrypoints/llm.py |
| 2 | Sampling Configuration | Sampling_Parameters | `SamplingParams(...)` | vllm/sampling_params.py |
| 3 | Prompt Preparation | Prompt_Formatting | Text/TokensPrompt | vllm/inputs/ |
| 4 | Batch Generation | Batch_Generation | `llm.generate()` | vllm/entrypoints/llm.py |
| 5 | Output Processing | Output_Processing | `RequestOutput` | vllm/outputs.py |

### Source Files (for enrichment)

- `vllm/__init__.py` - Main entry point API
- `vllm/sampling_params.py` - Comprehensive sampling control
- `vllm/outputs.py` - Generation output structures
- `examples/offline_inference/batch_llm_inference.py` - Large-scale batch processing
- `examples/offline_inference/automatic_prefix_caching.py` - Automatic prefix caching

### Step 1: LLM_Class_Initialization

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_LLM_Class_Initialization` |
| **Implementation** | `vllm-project_vllm_LLM_init` |
| **API Call** | `LLM(model: str, *, runner: RunnerOption = "auto", tokenizer: str | None = None, tokenizer_mode: TokenizerMode | str = "auto", skip_tokenizer_init: bool = False, trust_remote_code: bool = False, tensor_parallel_size: int = 1, dtype: ModelDType = "auto", quantization: QuantizationMethods | None = None, revision: str | None = None, seed: int = 0, gpu_memory_utilization: float = 0.9, swap_space: float = 4, enforce_eager: bool = False, **kwargs) -> LLM` |
| **Source Location** | `vllm/entrypoints/llm.py:L190-337` |
| **External Dependencies** | `torch`, `cloudpickle`, `pydantic`, `tqdm` |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `model: str` - HuggingFace model name or path, `tensor_parallel_size: int` - number of GPUs for TP, `dtype: str` - model precision (auto/float16/bfloat16), `gpu_memory_utilization: float` - GPU memory ratio for KV cache, `quantization: str` - quantization method (awq/gptq/fp8) |
| **Inputs** | Model identifier (HuggingFace name or local path), configuration parameters |
| **Outputs** | Initialized `LLM` instance with loaded model, tokenizer, and engine |

### Step 2: Sampling_Parameters

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Sampling_Parameters` |
| **Implementation** | `vllm-project_vllm_SamplingParams_init` |
| **API Call** | `SamplingParams(n: int = 1, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float = 0.0, seed: int | None = None, stop: str | list[str] | None = None, max_tokens: int | None = 16, logprobs: int | None = None, presence_penalty: float = 0.0, frequency_penalty: float = 0.0, repetition_penalty: float = 1.0) -> SamplingParams` |
| **Source Location** | `vllm/sampling_params.py:L111-241` |
| **External Dependencies** | `msgspec`, `pydantic` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `temperature: float` - sampling randomness (0=greedy), `top_p: float` - nucleus sampling threshold, `top_k: int` - top-k sampling, `max_tokens: int` - maximum tokens to generate, `stop: list[str]` - stop strings |
| **Inputs** | Sampling configuration values |
| **Outputs** | Configured `SamplingParams` instance controlling generation behavior |

### Step 3: Prompt_Formatting

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Prompt_Formatting` |
| **Implementation** | `vllm-project_vllm_PromptType_usage` |
| **API Call** | `TextPrompt(prompt: str) | TokensPrompt(prompt_token_ids: list[int]) | dict with "prompt" or "prompt_token_ids" keys` |
| **Source Location** | `vllm/inputs/__init__.py:L1-50` |
| **External Dependencies** | `typing` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `prompt: str` - raw text prompt, `prompt_token_ids: list[int]` - pre-tokenized prompt, `multi_modal_data: dict` - optional image/video data |
| **Inputs** | Raw text strings or pre-tokenized token IDs |
| **Outputs** | Formatted prompt acceptable by `LLM.generate()` |

### Step 4: Batch_Generation

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Batch_Generation` |
| **Implementation** | `vllm-project_vllm_LLM_generate` |
| **API Call** | `LLM.generate(prompts: PromptType | Sequence[PromptType], sampling_params: SamplingParams | Sequence[SamplingParams] | None = None, *, use_tqdm: bool | Callable = True, lora_request: LoRARequest | None = None, priority: list[int] | None = None) -> list[RequestOutput]` |
| **Source Location** | `vllm/entrypoints/llm.py:L365-434` |
| **External Dependencies** | `tqdm` |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `prompts: Sequence[PromptType]` - batch of prompts, `sampling_params: SamplingParams` - generation parameters, `use_tqdm: bool` - progress bar display, `lora_request: LoRARequest` - optional LoRA adapter |
| **Inputs** | Batch of prompts, sampling parameters, optional LoRA request |
| **Outputs** | List of `RequestOutput` objects containing generated sequences |

### Step 5: Output_Processing

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Output_Processing` |
| **Implementation** | `vllm-project_vllm_RequestOutput_usage` |
| **API Call** | `RequestOutput.outputs[i].text -> str` / `RequestOutput.outputs[i].token_ids -> list[int]` / `RequestOutput.outputs[i].logprobs -> SampleLogprobs` |
| **Source Location** | `vllm/outputs.py:L84-191` |
| **External Dependencies** | `torch` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `request_id: str` - unique request ID, `prompt: str` - original prompt, `outputs: list[CompletionOutput]` - generated completions, `finished: bool` - completion status |
| **Inputs** | `RequestOutput` objects from `LLM.generate()` |
| **Outputs** | Extracted text, token IDs, logprobs, and metadata from generation results |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| LLM_Class_Initialization | `LLM_init` | `LLM.__init__` | `vllm/entrypoints/llm.py:L190-337` | API Doc |
| Sampling_Parameters | `SamplingParams_init` | `SamplingParams` | `vllm/sampling_params.py:L111-241` | API Doc |
| Prompt_Formatting | `PromptType_usage` | `TextPrompt/TokensPrompt` | `vllm/inputs/__init__.py:L1-50` | Pattern Doc |
| Batch_Generation | `LLM_generate` | `LLM.generate` | `vllm/entrypoints/llm.py:L365-434` | API Doc |
| Output_Processing | `RequestOutput_usage` | `RequestOutput` | `vllm/outputs.py:L84-191` | API Doc |

---

## Workflow: vllm-project_vllm_OpenAI_Compatible_Serving

**File:** [→](./workflows/vllm-project_vllm_OpenAI_Compatible_Serving.md)
**Description:** Deploying vLLM as an OpenAI-compatible HTTP API server with Python client consumption.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Server Configuration | Server_Configuration | CLI args, env vars | vllm/entrypoints/openai/ |
| 2 | Server Launch | Server_Launch | `vllm serve` | vllm/entrypoints/openai/api_server.py |
| 3 | Client Initialization | OpenAI_Client_Setup | `OpenAI(base_url=...)` | openai package (external) |
| 4 | Chat Completion Request | Chat_Completion_API | `client.chat.completions.create()` | openai package (external) |
| 5 | Response Processing | Response_Handling | ChatCompletion response | openai package (external) |

### Source Files (for enrichment)

- `examples/online_serving/openai_chat_completion_client.py` - Basic OpenAI chat client
- `examples/online_serving/openai_chat_completion_client_with_tools.py` - Tool calling streaming
- `examples/online_serving/gradio_openai_chatbot_webserver.py` - Gradio chatbot OpenAI API
- `examples/online_serving/streamlit_openai_chatbot_webserver.py` - Advanced Streamlit chatbot
- `benchmarks/multi_turn/benchmark_serving_multi_turn.py` - Conversational serving perf

### Step 1: Server_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Server_Configuration` |
| **Implementation** | `vllm-project_vllm_vllm_serve_args` |
| **API Call** | `vllm serve <model> [--host HOST] [--port PORT] [--tensor-parallel-size N] [--dtype TYPE] [--max-model-len N] [--api-key KEY] [--enable-lora] [--chat-template TEMPLATE]` |
| **Source Location** | `vllm/entrypoints/openai/cli_args.py:L1-200` (CLI args), `vllm/engine/arg_utils.py:L1-300` (EngineArgs) |
| **External Dependencies** | `argparse`, `uvicorn`, `fastapi` |
| **Environment** | `vllm-project_vllm_Server_Environment` |
| **Key Parameters** | `--host: str` - bind address (default 0.0.0.0), `--port: int` - port number (default 8000), `--tensor-parallel-size: int` - TP degree, `--api-key: str` - optional authentication |
| **Inputs** | Command-line arguments, environment variables |
| **Outputs** | Configured server settings for API deployment |

### Step 2: Server_Launch

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Server_Launch` |
| **Implementation** | `vllm-project_vllm_api_server_run` |
| **API Call** | `vllm serve <model_name>` (CLI) or `uvicorn vllm.entrypoints.openai.api_server:app` |
| **Source Location** | `vllm/entrypoints/openai/api_server.py:L1-500` |
| **External Dependencies** | `uvicorn`, `fastapi`, `uvloop` |
| **Environment** | `vllm-project_vllm_Server_Environment` |
| **Key Parameters** | `model: str` - HuggingFace model path, `host: str` - server bind address, `port: int` - server port |
| **Inputs** | Model path and server configuration |
| **Outputs** | Running HTTP server exposing OpenAI-compatible endpoints at `/v1/chat/completions`, `/v1/completions`, `/v1/models` |

### Step 3: OpenAI_Client_Setup

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_OpenAI_Client_Setup` |
| **Implementation** | `vllm-project_vllm_OpenAI_client_init` |
| **API Call** | `OpenAI(api_key: str = "EMPTY", base_url: str = "http://localhost:8000/v1") -> OpenAI` |
| **Source Location** | External: `openai` package |
| **External Dependencies** | `openai>=1.0.0` |
| **Environment** | `vllm-project_vllm_Client_Environment` |
| **Key Parameters** | `api_key: str` - API key (use "EMPTY" for no auth), `base_url: str` - vLLM server URL with /v1 suffix |
| **Inputs** | vLLM server URL and optional API key |
| **Outputs** | Configured `OpenAI` client instance pointing to vLLM server |

### Step 4: Chat_Completion_API

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Chat_Completion_API` |
| **Implementation** | `vllm-project_vllm_chat_completions_create` |
| **API Call** | `client.chat.completions.create(model: str, messages: list[dict], temperature: float = 1.0, max_tokens: int | None = None, stream: bool = False, tools: list[dict] | None = None) -> ChatCompletion | Stream[ChatCompletionChunk]` |
| **Source Location** | External: `openai` package; Server: `vllm/entrypoints/openai/serving_chat.py:L1-400` |
| **External Dependencies** | `openai>=1.0.0` |
| **Environment** | `vllm-project_vllm_Client_Environment` |
| **Key Parameters** | `model: str` - model name, `messages: list[dict]` - conversation history with role/content, `stream: bool` - enable streaming, `tools: list` - function calling definitions |
| **Inputs** | Model name, message history, generation parameters |
| **Outputs** | `ChatCompletion` response with generated assistant message or streaming chunks |

### Step 5: Response_Handling

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Response_Handling` |
| **Implementation** | `vllm-project_vllm_ChatCompletion_processing` |
| **API Call** | `response.choices[0].message.content -> str` / `response.choices[0].message.tool_calls -> list` / `response.usage.total_tokens -> int` |
| **Source Location** | External: `openai` package types |
| **External Dependencies** | `openai>=1.0.0` |
| **Environment** | `vllm-project_vllm_Client_Environment` |
| **Key Parameters** | `choices: list` - generated completions, `usage: Usage` - token counts, `message.content: str` - generated text, `message.tool_calls: list` - function calls |
| **Inputs** | `ChatCompletion` or streamed `ChatCompletionChunk` objects |
| **Outputs** | Extracted generated text, tool calls, usage statistics |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Server_Configuration | `vllm_serve_args` | CLI args | `vllm/entrypoints/openai/cli_args.py` | External Tool Doc |
| Server_Launch | `api_server_run` | `vllm serve` | `vllm/entrypoints/openai/api_server.py` | External Tool Doc |
| OpenAI_Client_Setup | `OpenAI_client_init` | `OpenAI()` | openai package | Wrapper Doc |
| Chat_Completion_API | `chat_completions_create` | `chat.completions.create` | openai package | Wrapper Doc |
| Response_Handling | `ChatCompletion_processing` | `ChatCompletion` | openai package | Pattern Doc |

---

## Workflow: vllm-project_vllm_Multi_LoRA_Inference

**File:** [→](./workflows/vllm-project_vllm_Multi_LoRA_Inference.md)
**Description:** Serving multiple LoRA adapters dynamically on a single base model.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Engine Configuration with LoRA | LoRA_Engine_Configuration | `EngineArgs(enable_lora=True)` | vllm/engine/arg_utils.py |
| 2 | LoRA Adapter Registration | LoRA_Adapter_Registration | `LoRARequest(name, id, path)` | vllm/lora/request.py |
| 3 | Request Submission with Adapter | LoRA_Request_Submission | `engine.add_request(..., lora_request)` | vllm/engine/llm_engine.py |
| 4 | Adapter-Aware Scheduling | LoRA_Scheduling | Scheduler internals | vllm/core/scheduler.py |
| 5 | Output Processing | LoRA_Output_Processing | `RequestOutput` | vllm/outputs.py |

### Source Files (for enrichment)

- `examples/offline_inference/multilora_inference.py` - Multi-adapter serving
- `examples/offline_inference/lora_with_quantization_inference.py` - LoRA on quantized models
- `benchmarks/kernels/benchmark_lora.py` - LoRA adapter operations

### Step 1: LoRA_Engine_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_LoRA_Engine_Configuration` |
| **Implementation** | `vllm-project_vllm_EngineArgs_lora` |
| **API Call** | `EngineArgs(model: str, enable_lora: bool = True, max_loras: int = 1, max_lora_rank: int = 8, max_cpu_loras: int | None = None, lora_extra_vocab_size: int = 256) -> EngineArgs` |
| **Source Location** | `vllm/engine/arg_utils.py:L1-300` |
| **External Dependencies** | `pydantic`, `huggingface_hub` |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `enable_lora: bool` - enable LoRA support, `max_loras: int` - concurrent adapters in batch, `max_lora_rank: int` - maximum supported rank, `max_cpu_loras: int` - CPU cache size |
| **Inputs** | Base model path, LoRA configuration parameters |
| **Outputs** | Configured `EngineArgs` with LoRA support enabled |

### Step 2: LoRA_Adapter_Registration

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_LoRA_Adapter_Registration` |
| **Implementation** | `vllm-project_vllm_LoRARequest_init` |
| **API Call** | `LoRARequest(lora_name: str, lora_int_id: int, lora_path: str, long_lora_max_len: int | None = None, base_model_name: str | None = None) -> LoRARequest` |
| **Source Location** | `vllm/lora/request.py:L9-96` |
| **External Dependencies** | `msgspec` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `lora_name: str` - unique adapter name, `lora_int_id: int` - unique integer ID (>0), `lora_path: str` - path to adapter weights |
| **Inputs** | LoRA adapter name, ID, and path to weights |
| **Outputs** | `LoRARequest` instance identifying adapter for requests |

### Step 3: LoRA_Request_Submission

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_LoRA_Request_Submission` |
| **Implementation** | `vllm-project_vllm_LLMEngine_add_request_lora` |
| **API Call** | `engine.add_request(request_id: str, prompt: str, sampling_params: SamplingParams, lora_request: LoRARequest | None = None) -> None` |
| **Source Location** | `vllm/v1/engine/llm_engine.py:L100-200` |
| **External Dependencies** | None |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `request_id: str` - unique request identifier, `prompt: str` - input prompt, `lora_request: LoRARequest` - adapter to use |
| **Inputs** | Request ID, prompt, sampling params, LoRA request |
| **Outputs** | Request queued for processing with specified adapter |

### Step 4: LoRA_Scheduling

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_LoRA_Scheduling` |
| **Implementation** | `vllm-project_vllm_Scheduler_lora_batching` |
| **API Call** | Internal: Scheduler groups requests by LoRA adapter, respecting `max_loras` limit per batch |
| **Source Location** | `vllm/core/scheduler.py:L1-500` |
| **External Dependencies** | None |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `max_loras: int` - maximum adapters per batch (from EngineArgs) |
| **Inputs** | Pending requests with LoRA associations |
| **Outputs** | Batched requests optimized for adapter switching |

### Step 5: LoRA_Output_Processing

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_LoRA_Output_Processing` |
| **Implementation** | `vllm-project_vllm_RequestOutput_lora` |
| **API Call** | `RequestOutput.lora_request -> LoRARequest | None` / `CompletionOutput.lora_request -> LoRARequest | None` |
| **Source Location** | `vllm/outputs.py:L23-63` (CompletionOutput), `vllm/outputs.py:L84-191` (RequestOutput) |
| **External Dependencies** | None |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `lora_request: LoRARequest | None` - adapter used for this output |
| **Inputs** | `RequestOutput` objects from engine |
| **Outputs** | Generated text with LoRA adapter attribution |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| LoRA_Engine_Configuration | `EngineArgs_lora` | `EngineArgs` | `vllm/engine/arg_utils.py:L1-300` | API Doc |
| LoRA_Adapter_Registration | `LoRARequest_init` | `LoRARequest` | `vllm/lora/request.py:L9-96` | API Doc |
| LoRA_Request_Submission | `LLMEngine_add_request_lora` | `add_request` | `vllm/v1/engine/llm_engine.py:L100-200` | API Doc |
| LoRA_Scheduling | `Scheduler_lora_batching` | Scheduler | `vllm/core/scheduler.py` | Pattern Doc |
| LoRA_Output_Processing | `RequestOutput_lora` | `RequestOutput` | `vllm/outputs.py:L84-191` | API Doc |

---

## Workflow: vllm-project_vllm_Vision_Language_Inference

**File:** [→](./workflows/vllm-project_vllm_Vision_Language_Inference.md)
**Description:** Running inference on vision-language models (VLMs) with image and text inputs.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | VLM Model Configuration | VLM_Model_Configuration | `EngineArgs(limit_mm_per_prompt=...)` | vllm/engine/arg_utils.py |
| 2 | Image Input Preparation | Image_Input_Preparation | PIL Image, URLs, base64 | vllm/multimodal/ |
| 3 | Prompt Construction with Image Tokens | VLM_Prompt_Construction | `<image>` placeholders | vllm/inputs/ |
| 4 | Multimodal Generation | Multimodal_Generation | `llm.generate(multi_modal_data=...)` | vllm/entrypoints/llm.py |
| 5 | VLM Output Processing | VLM_Output_Processing | `RequestOutput` | vllm/outputs.py |

### Source Files (for enrichment)

- `examples/offline_inference/vision_language.py` - VLM comprehensive reference (2243 lines)
- `examples/offline_inference/vision_language_multi_image.py` - Multi-image VLM reference
- `examples/offline_inference/encoder_decoder_multimodal.py` - Encoder-decoder models
- `examples/pooling/pooling/vision_language_pooling.py` - Vision-language embeddings
- `examples/online_serving/openai_chat_completion_client_for_multimodal.py` - Multimodal inputs client

### Step 1: VLM_Model_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_VLM_Model_Configuration` |
| **Implementation** | `vllm-project_vllm_EngineArgs_vlm` |
| **API Call** | `EngineArgs(model: str, limit_mm_per_prompt: dict[str, int] = {"image": 1}, max_model_len: int = None, mm_processor_kwargs: dict = None) -> EngineArgs` or `LLM(model: str, limit_mm_per_prompt: dict = {"image": 1}, ...)` |
| **Source Location** | `vllm/engine/arg_utils.py:L1-300`, `vllm/entrypoints/llm.py:L190-337` |
| **External Dependencies** | `transformers`, `PIL` |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `limit_mm_per_prompt: dict` - max images/videos per prompt, `mm_processor_kwargs: dict` - model-specific processor args |
| **Inputs** | VLM model path, multimodal limits, processor configuration |
| **Outputs** | Configured engine/LLM with multimodal support |

### Step 2: Image_Input_Preparation

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Image_Input_Preparation` |
| **Implementation** | `vllm-project_vllm_MultiModalData_image` |
| **API Call** | `PIL.Image.open(path) -> Image` / `{"image": Image | list[Image] | str (URL)}` |
| **Source Location** | `vllm/multimodal/image.py:L1-100` |
| **External Dependencies** | `PIL`, `requests` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `image: PIL.Image` - loaded image, `url: str` - image URL for remote loading |
| **Inputs** | Image file paths, URLs, or PIL Image objects |
| **Outputs** | Preprocessed image data for prompt construction |

### Step 3: VLM_Prompt_Construction

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_VLM_Prompt_Construction` |
| **Implementation** | `vllm-project_vllm_VLM_prompt_format` |
| **API Call** | `{"prompt": str_with_image_placeholders, "multi_modal_data": {"image": Image | list[Image]}}` |
| **Source Location** | `vllm/inputs/__init__.py`, Model-specific in `examples/offline_inference/vision_language.py:L42-200` |
| **External Dependencies** | `transformers` (for chat templates) |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `prompt: str` - text with `<image>` or model-specific placeholders, `multi_modal_data: dict` - image data keyed by modality |
| **Inputs** | Text prompt template, image data |
| **Outputs** | Combined prompt dict with text and multimodal data |

### Step 4: Multimodal_Generation

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Multimodal_Generation` |
| **Implementation** | `vllm-project_vllm_LLM_generate_mm` |
| **API Call** | `llm.generate(prompts: list[dict], sampling_params: SamplingParams) -> list[RequestOutput]` where prompts contain `multi_modal_data` |
| **Source Location** | `vllm/entrypoints/llm.py:L365-434` |
| **External Dependencies** | `torch`, `transformers` |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `prompts: list[dict]` - prompts with `multi_modal_data` field, `sampling_params: SamplingParams` - generation config |
| **Inputs** | Multimodal prompts with images, sampling parameters |
| **Outputs** | `RequestOutput` objects with generated text descriptions |

### Step 5: VLM_Output_Processing

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_VLM_Output_Processing` |
| **Implementation** | `vllm-project_vllm_RequestOutput_vlm` |
| **API Call** | `RequestOutput.outputs[0].text -> str` / `RequestOutput.multi_modal_placeholders -> dict` |
| **Source Location** | `vllm/outputs.py:L84-191` |
| **External Dependencies** | None |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `outputs: list[CompletionOutput]` - generated text, `multi_modal_placeholders: dict` - placeholder positions |
| **Inputs** | `RequestOutput` from VLM generation |
| **Outputs** | Generated text and multimodal metadata |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| VLM_Model_Configuration | `EngineArgs_vlm` | `EngineArgs/LLM` | `vllm/engine/arg_utils.py`, `vllm/entrypoints/llm.py` | API Doc |
| Image_Input_Preparation | `MultiModalData_image` | `PIL.Image` | `vllm/multimodal/image.py` | Pattern Doc |
| VLM_Prompt_Construction | `VLM_prompt_format` | dict prompt | `examples/offline_inference/vision_language.py` | Pattern Doc |
| Multimodal_Generation | `LLM_generate_mm` | `LLM.generate` | `vllm/entrypoints/llm.py:L365-434` | API Doc |
| VLM_Output_Processing | `RequestOutput_vlm` | `RequestOutput` | `vllm/outputs.py:L84-191` | API Doc |

---

## Workflow: vllm-project_vllm_Speculative_Decoding

**File:** [→](./workflows/vllm-project_vllm_Speculative_Decoding.md)
**Description:** Accelerating inference using speculative decoding techniques (EAGLE, n-gram, MTP).

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Speculative Method Selection | Speculative_Method_Selection | "eagle", "ngram", "mtp" | — |
| 2 | Speculative Configuration | Speculative_Configuration | `speculative_config={...}` | vllm/config.py |
| 3 | Engine Initialization | Speculative_Engine_Init | `LLM(speculative_config=...)` | vllm/entrypoints/llm.py |
| 4 | Speculative Generation | Speculative_Generation | `llm.generate()` | vllm/entrypoints/llm.py |
| 5 | Acceptance Metrics Analysis | Speculation_Metrics | `llm.get_metrics()` | vllm/v1/metrics/ |

### Source Files (for enrichment)

- `examples/offline_inference/spec_decode.py` - Speculative decoding example
- `examples/offline_inference/mlpspeculator.py` - MLP speculative decoding
- `benchmarks/benchmark_ngram_proposer.py` - N-gram speculative decoding

### Step 1: Speculative_Method_Selection

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Speculative_Method_Selection` |
| **Implementation** | `vllm-project_vllm_SpeculativeMethod_choice` |
| **API Call** | `method: Literal["eagle", "eagle3", "ngram", "mtp", "medusa", "mlp_speculator", "draft_model", "suffix"]` |
| **Source Location** | `vllm/config/speculative.py:L42-49` |
| **External Dependencies** | None |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `method: str` - speculative method type |
| **Inputs** | Method name selection |
| **Outputs** | Selected speculative decoding strategy |

### Step 2: Speculative_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Speculative_Configuration` |
| **Implementation** | `vllm-project_vllm_SpeculativeConfig_init` |
| **API Call** | `speculative_config={"method": str, "model": str | None, "num_speculative_tokens": int, "prompt_lookup_max": int | None, "prompt_lookup_min": int | None}` |
| **Source Location** | `vllm/config/speculative.py:L54-150` |
| **External Dependencies** | `pydantic` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `method: str` - decoding method, `model: str` - draft model path (for eagle/draft_model), `num_speculative_tokens: int` - tokens to speculate, `prompt_lookup_max/min: int` - ngram window sizes |
| **Inputs** | Speculative method configuration dict |
| **Outputs** | Configured `SpeculativeConfig` for engine initialization |

### Step 3: Speculative_Engine_Init

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Speculative_Engine_Init` |
| **Implementation** | `vllm-project_vllm_LLM_speculative` |
| **API Call** | `LLM(model: str, speculative_config: dict, tensor_parallel_size: int = 1, enforce_eager: bool = False, ...) -> LLM` |
| **Source Location** | `vllm/entrypoints/llm.py:L190-337` |
| **External Dependencies** | `torch`, `transformers` |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `model: str` - target model, `speculative_config: dict` - speculation configuration |
| **Inputs** | Target model, speculative configuration |
| **Outputs** | LLM instance with speculative decoding enabled |

### Step 4: Speculative_Generation

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Speculative_Generation` |
| **Implementation** | `vllm-project_vllm_LLM_generate_spec` |
| **API Call** | `llm.generate(prompts: Sequence[PromptType], sampling_params: SamplingParams) -> list[RequestOutput]` |
| **Source Location** | `vllm/entrypoints/llm.py:L365-434` |
| **External Dependencies** | None |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | Same as regular generate - speculative decoding is transparent |
| **Inputs** | Prompts and sampling parameters |
| **Outputs** | `RequestOutput` objects (same API, accelerated internally) |

### Step 5: Speculation_Metrics

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Speculation_Metrics` |
| **Implementation** | `vllm-project_vllm_get_metrics_spec` |
| **API Call** | `llm.get_metrics() -> list[Metric]` with `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_accepted_tokens`, `vllm:spec_decode_num_accepted_tokens_per_pos` |
| **Source Location** | `vllm/entrypoints/llm.py:L1539-1549`, `vllm/v1/metrics/reader.py` |
| **External Dependencies** | `prometheus_client` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | Returns `Counter` for draft/accepted counts, `Vector` for per-position acceptance |
| **Inputs** | None (reads from internal metrics) |
| **Outputs** | List of `Metric` objects with speculation statistics |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Speculative_Method_Selection | `SpeculativeMethod_choice` | `Literal` type | `vllm/config/speculative.py:L42-49` | Pattern Doc |
| Speculative_Configuration | `SpeculativeConfig_init` | `speculative_config` dict | `vllm/config/speculative.py:L54-150` | API Doc |
| Speculative_Engine_Init | `LLM_speculative` | `LLM.__init__` | `vllm/entrypoints/llm.py:L190-337` | API Doc |
| Speculative_Generation | `LLM_generate_spec` | `LLM.generate` | `vllm/entrypoints/llm.py:L365-434` | API Doc |
| Speculation_Metrics | `get_metrics_spec` | `LLM.get_metrics` | `vllm/entrypoints/llm.py:L1539-1549` | API Doc |

---

## Workflow: vllm-project_vllm_Structured_Output_Generation

**File:** [→](./workflows/vllm-project_vllm_Structured_Output_Generation.md)
**Description:** Generating constrained outputs using JSON schemas, regex, grammars, or choices.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Constraint Definition | Constraint_Definition | json/regex/choice/grammar | — |
| 2 | StructuredOutputsParams Configuration | StructuredOutputsParams | `StructuredOutputsParams(json=...)` | vllm/sampling_params.py |
| 3 | SamplingParams Integration | Structured_SamplingParams | `SamplingParams(structured_outputs=...)` | vllm/sampling_params.py |
| 4 | Constrained Generation | Constrained_Generation | `llm.generate()` with logit masking | vllm/entrypoints/llm.py |
| 5 | Structured Output Parsing | Structured_Output_Parsing | `json.loads()`, direct use | — |

### Source Files (for enrichment)

- `examples/offline_inference/structured_outputs.py` - JSON schema constraints
- `vllm/sampling_params.py` - StructuredOutputsParams definition
- `benchmarks/benchmark_serving_structured_output.py` - Structured output constraints

### Step 1: Constraint_Definition

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Constraint_Definition` |
| **Implementation** | `vllm-project_vllm_StructuredOutputsParams_types` |
| **API Call** | JSON schema: `pydantic.BaseModel.model_json_schema()` / Regex: `r"pattern"` / Choice: `["opt1", "opt2"]` / Grammar: EBNF string |
| **Source Location** | `vllm/sampling_params.py:L32-99`, `examples/offline_inference/structured_outputs.py:L43-84` |
| **External Dependencies** | `pydantic` (for JSON schema) |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `json: dict` - JSON schema, `regex: str` - regex pattern, `choice: list[str]` - allowed values, `grammar: str` - EBNF grammar |
| **Inputs** | Constraint specification (one of json/regex/choice/grammar) |
| **Outputs** | Constraint definition for StructuredOutputsParams |

### Step 2: StructuredOutputsParams_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_StructuredOutputsParams_Configuration` |
| **Implementation** | `vllm-project_vllm_StructuredOutputsParams_init` |
| **API Call** | `StructuredOutputsParams(json: str | dict | None = None, regex: str | None = None, choice: list[str] | None = None, grammar: str | None = None, json_object: bool | None = None, disable_fallback: bool = False) -> StructuredOutputsParams` |
| **Source Location** | `vllm/sampling_params.py:L32-99` |
| **External Dependencies** | `pydantic` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `json: dict` - JSON schema constraint, `regex: str` - regex pattern, `choice: list[str]` - allowed outputs, `grammar: str` - EBNF grammar, `disable_fallback: bool` - strict mode |
| **Inputs** | One constraint type (mutually exclusive) |
| **Outputs** | Configured `StructuredOutputsParams` instance |

### Step 3: Structured_SamplingParams

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Structured_SamplingParams` |
| **Implementation** | `vllm-project_vllm_SamplingParams_structured` |
| **API Call** | `SamplingParams(structured_outputs: StructuredOutputsParams, max_tokens: int = 50, ...) -> SamplingParams` |
| **Source Location** | `vllm/sampling_params.py:L111-241` |
| **External Dependencies** | `msgspec` |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `structured_outputs: StructuredOutputsParams` - constraint configuration |
| **Inputs** | StructuredOutputsParams and other sampling parameters |
| **Outputs** | `SamplingParams` with structured output constraints |

### Step 4: Constrained_Generation

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Constrained_Generation` |
| **Implementation** | `vllm-project_vllm_LLM_generate_structured` |
| **API Call** | `llm.generate(prompts: Sequence[PromptType], sampling_params: SamplingParams) -> list[RequestOutput]` |
| **Source Location** | `vllm/entrypoints/llm.py:L365-434`, `vllm/v1/sample/logits_processor.py` |
| **External Dependencies** | `outlines` or `lm-format-enforcer` (backend-dependent) |
| **Environment** | `vllm-project_vllm_GPU_Environment` |
| **Key Parameters** | `sampling_params: SamplingParams` - with `structured_outputs` set |
| **Inputs** | Prompts and constrained sampling parameters |
| **Outputs** | `RequestOutput` with text conforming to constraints |

### Step 5: Structured_Output_Parsing

| Attribute | Value |
|-----------|-------|
| **Principle** | `vllm-project_vllm_Structured_Output_Parsing` |
| **Implementation** | `vllm-project_vllm_structured_output_parse` |
| **API Call** | `json.loads(output.outputs[0].text) -> dict` (for JSON) / direct string use (for regex/choice/grammar) |
| **Source Location** | Python stdlib `json`, `examples/offline_inference/structured_outputs.py:L91-93` |
| **External Dependencies** | `json` (stdlib) |
| **Environment** | `vllm-project_vllm_Python_Environment` |
| **Key Parameters** | `text: str` - generated output text |
| **Inputs** | Generated text from constrained output |
| **Outputs** | Parsed structured data (dict for JSON, string for others) |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Constraint_Definition | `StructuredOutputsParams_types` | Pydantic/regex/grammar | `vllm/sampling_params.py:L32-99` | Pattern Doc |
| StructuredOutputsParams_Configuration | `StructuredOutputsParams_init` | `StructuredOutputsParams` | `vllm/sampling_params.py:L32-99` | API Doc |
| Structured_SamplingParams | `SamplingParams_structured` | `SamplingParams` | `vllm/sampling_params.py:L111-241` | API Doc |
| Constrained_Generation | `LLM_generate_structured` | `LLM.generate` | `vllm/entrypoints/llm.py:L365-434` | API Doc |
| Structured_Output_Parsing | `structured_output_parse` | `json.loads` | stdlib | Pattern Doc |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

**Implementation Types:**
- **API Doc:** Function/class in this repo
- **Wrapper Doc:** External library with repo-specific usage
- **Pattern Doc:** User-defined interface/pattern
- **External Tool Doc:** CLI or external tool
