# Implementation Index: vllm-project_vllm

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying a Implementation page.

## Summary

| Workflow | Implementations | Count |
|----------|-----------------|-------|
| Basic_Offline_Inference | LLM_init, SamplingParams_init, PromptType_usage, LLM_generate, RequestOutput_usage | 5 |
| OpenAI_Compatible_Serving | vllm_serve_args, api_server_run, OpenAI_client_init, chat_completions_create, ChatCompletion_processing | 5 |
| Multi_LoRA_Inference | EngineArgs_lora, LoRARequest_init, LLMEngine_add_request_lora, Scheduler_lora_batching, RequestOutput_lora | 5 |
| Vision_Language_Inference | EngineArgs_vlm, MultiModalData_image, VLM_prompt_format, LLM_generate_mm, RequestOutput_vlm | 5 |
| Speculative_Decoding | SpeculativeMethod_choice, SpeculativeConfig_init, LLM_speculative, LLM_generate_spec, get_metrics_spec | 5 |
| Structured_Output_Generation | StructuredOutputsParams_types, StructuredOutputsParams_init, SamplingParams_structured, LLM_generate_structured, structured_output_parse | 5 |
| **Total** | | **30** |

---

## Pages

### Basic_Offline_Inference

| Page | File | Connections | Type |
|------|------|-------------|------|
| vllm-project_vllm_LLM_init | implementations/vllm-project_vllm_LLM_init.md | ✅Principle:LLM_Class_Initialization, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_SamplingParams_init | implementations/vllm-project_vllm_SamplingParams_init.md | ✅Principle:Sampling_Parameters, ✅Env:vllm-project_vllm_Python_Environment | API Doc |
| vllm-project_vllm_PromptType_usage | implementations/vllm-project_vllm_PromptType_usage.md | ✅Principle:Prompt_Formatting, ✅Env:vllm-project_vllm_Python_Environment | Pattern Doc |
| vllm-project_vllm_LLM_generate | implementations/vllm-project_vllm_LLM_generate.md | ✅Principle:Batch_Generation, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_RequestOutput_usage | implementations/vllm-project_vllm_RequestOutput_usage.md | ✅Principle:Output_Processing, ✅Env:vllm-project_vllm_Python_Environment | API Doc |

### OpenAI_Compatible_Serving

| Page | File | Connections | Type |
|------|------|-------------|------|
| vllm-project_vllm_vllm_serve_args | implementations/vllm-project_vllm_vllm_serve_args.md | ✅Principle:Server_Configuration, ✅Env:vllm-project_vllm_Server_Environment | External Tool Doc |
| vllm-project_vllm_api_server_run | implementations/vllm-project_vllm_api_server_run.md | ✅Principle:Server_Launch, ✅Env:vllm-project_vllm_Server_Environment | External Tool Doc |
| vllm-project_vllm_OpenAI_client_init | implementations/vllm-project_vllm_OpenAI_client_init.md | ✅Principle:OpenAI_Client_Setup, ✅Env:vllm-project_vllm_Client_Environment | Wrapper Doc |
| vllm-project_vllm_chat_completions_create | implementations/vllm-project_vllm_chat_completions_create.md | ✅Principle:Chat_Completion_API, ✅Env:vllm-project_vllm_Client_Environment | Wrapper Doc |
| vllm-project_vllm_ChatCompletion_processing | implementations/vllm-project_vllm_ChatCompletion_processing.md | ✅Principle:Response_Handling, ✅Env:vllm-project_vllm_Client_Environment | Pattern Doc |

### Multi_LoRA_Inference

| Page | File | Connections | Type |
|------|------|-------------|------|
| vllm-project_vllm_EngineArgs_lora | implementations/vllm-project_vllm_EngineArgs_lora.md | ✅Principle:LoRA_Engine_Configuration, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_LoRARequest_init | implementations/vllm-project_vllm_LoRARequest_init.md | ✅Principle:LoRA_Adapter_Registration, ✅Env:vllm-project_vllm_Python_Environment | API Doc |
| vllm-project_vllm_LLMEngine_add_request_lora | implementations/vllm-project_vllm_LLMEngine_add_request_lora.md | ✅Principle:LoRA_Request_Submission, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_Scheduler_lora_batching | implementations/vllm-project_vllm_Scheduler_lora_batching.md | ✅Principle:LoRA_Scheduling, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_RequestOutput_lora | implementations/vllm-project_vllm_RequestOutput_lora.md | ✅Principle:LoRA_Output_Processing, ✅Env:vllm-project_vllm_Python_Environment | API Doc |

### Vision_Language_Inference

| Page | File | Connections | Type |
|------|------|-------------|------|
| vllm-project_vllm_EngineArgs_vlm | implementations/vllm-project_vllm_EngineArgs_vlm.md | ✅Principle:VLM_Model_Configuration, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_MultiModalData_image | implementations/vllm-project_vllm_MultiModalData_image.md | ✅Principle:Image_Input_Preparation, ✅Env:vllm-project_vllm_Python_Environment | Pattern Doc |
| vllm-project_vllm_VLM_prompt_format | implementations/vllm-project_vllm_VLM_prompt_format.md | ✅Principle:VLM_Prompt_Construction, ✅Env:vllm-project_vllm_Python_Environment | Pattern Doc |
| vllm-project_vllm_LLM_generate_mm | implementations/vllm-project_vllm_LLM_generate_mm.md | ✅Principle:Multimodal_Generation, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_RequestOutput_vlm | implementations/vllm-project_vllm_RequestOutput_vlm.md | ✅Principle:VLM_Output_Processing, ✅Env:vllm-project_vllm_Python_Environment | API Doc |

### Speculative_Decoding

| Page | File | Connections | Type |
|------|------|-------------|------|
| vllm-project_vllm_SpeculativeMethod_choice | implementations/vllm-project_vllm_SpeculativeMethod_choice.md | ✅Principle:Speculative_Method_Selection, ✅Env:vllm-project_vllm_Python_Environment | Pattern Doc |
| vllm-project_vllm_SpeculativeConfig_init | implementations/vllm-project_vllm_SpeculativeConfig_init.md | ✅Principle:Speculative_Configuration, ✅Env:vllm-project_vllm_Python_Environment | API Doc |
| vllm-project_vllm_LLM_speculative | implementations/vllm-project_vllm_LLM_speculative.md | ✅Principle:Speculative_Engine_Init, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_LLM_generate_spec | implementations/vllm-project_vllm_LLM_generate_spec.md | ✅Principle:Speculative_Generation, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_get_metrics_spec | implementations/vllm-project_vllm_get_metrics_spec.md | ✅Principle:Speculation_Metrics, ✅Env:vllm-project_vllm_Python_Environment | API Doc |

### Structured_Output_Generation

| Page | File | Connections | Type |
|------|------|-------------|------|
| vllm-project_vllm_StructuredOutputsParams_types | implementations/vllm-project_vllm_StructuredOutputsParams_types.md | ✅Principle:Constraint_Definition, ✅Env:vllm-project_vllm_Python_Environment | Pattern Doc |
| vllm-project_vllm_StructuredOutputsParams_init | implementations/vllm-project_vllm_StructuredOutputsParams_init.md | ✅Principle:StructuredOutputsParams_Configuration, ✅Env:vllm-project_vllm_Python_Environment | API Doc |
| vllm-project_vllm_SamplingParams_structured | implementations/vllm-project_vllm_SamplingParams_structured.md | ✅Principle:Structured_SamplingParams, ✅Env:vllm-project_vllm_Python_Environment | API Doc |
| vllm-project_vllm_LLM_generate_structured | implementations/vllm-project_vllm_LLM_generate_structured.md | ✅Principle:Constrained_Generation, ✅Env:vllm-project_vllm_GPU_Environment | API Doc |
| vllm-project_vllm_structured_output_parse | implementations/vllm-project_vllm_structured_output_parse.md | ✅Principle:Structured_Output_Parsing, ✅Env:vllm-project_vllm_Python_Environment | Pattern Doc |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

**Implementation Types:**
- **API Doc:** Function/class in this repo
- **Wrapper Doc:** External library with repo-specific usage
- **Pattern Doc:** User-defined interface/pattern
- **External Tool Doc:** CLI or external tool
