# Principle Index: vllm-project_vllm

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Summary

| Workflow | Principles | Count |
|----------|------------|-------|
| Basic_Offline_Inference | LLM_Class_Initialization, Sampling_Parameters, Prompt_Formatting, Batch_Generation, Output_Processing | 5 |
| OpenAI_Compatible_Serving | Server_Configuration, Server_Launch, OpenAI_Client_Setup, Chat_Completion_API, Response_Handling | 5 |
| Multi_LoRA_Inference | LoRA_Engine_Configuration, LoRA_Adapter_Registration, LoRA_Request_Submission, LoRA_Scheduling, LoRA_Output_Processing | 5 |
| Vision_Language_Inference | VLM_Model_Configuration, Image_Input_Preparation, VLM_Prompt_Construction, Multimodal_Generation, VLM_Output_Processing | 5 |
| Speculative_Decoding | Speculative_Method_Selection, Speculative_Configuration, Speculative_Engine_Init, Speculative_Generation, Speculation_Metrics | 5 |
| Structured_Output_Generation | Constraint_Definition, StructuredOutputsParams_Configuration, Structured_SamplingParams, Constrained_Generation, Structured_Output_Parsing | 5 |
| **Total** | | **30** |

---

## Pages

### Basic_Offline_Inference

| Page | File | Connections | Domains |
|------|------|-------------|---------|
| vllm-project_vllm_LLM_Class_Initialization | principles/vllm-project_vllm_LLM_Class_Initialization.md | ✅Impl:LLM_init, ✅Workflow:Basic_Offline_Inference | NLP, Inference, Configuration |
| vllm-project_vllm_Sampling_Parameters | principles/vllm-project_vllm_Sampling_Parameters.md | ✅Impl:SamplingParams_init, ✅Workflow:Basic_Offline_Inference | NLP, Inference, Sampling |
| vllm-project_vllm_Prompt_Formatting | principles/vllm-project_vllm_Prompt_Formatting.md | ✅Impl:PromptType_usage, ✅Workflow:Basic_Offline_Inference | NLP, Input_Processing |
| vllm-project_vllm_Batch_Generation | principles/vllm-project_vllm_Batch_Generation.md | ✅Impl:LLM_generate, ✅Workflow:Basic_Offline_Inference | NLP, Inference, Batching |
| vllm-project_vllm_Output_Processing | principles/vllm-project_vllm_Output_Processing.md | ✅Impl:RequestOutput_usage, ✅Workflow:Basic_Offline_Inference | NLP, Output_Processing |

### OpenAI_Compatible_Serving

| Page | File | Connections | Domains |
|------|------|-------------|---------|
| vllm-project_vllm_Server_Configuration | principles/vllm-project_vllm_Server_Configuration.md | ✅Impl:vllm_serve_args, ✅Workflow:OpenAI_Compatible_Serving | Serving, Configuration, API |
| vllm-project_vllm_Server_Launch | principles/vllm-project_vllm_Server_Launch.md | ✅Impl:api_server_run, ✅Workflow:OpenAI_Compatible_Serving | Serving, Deployment |
| vllm-project_vllm_OpenAI_Client_Setup | principles/vllm-project_vllm_OpenAI_Client_Setup.md | ✅Impl:OpenAI_client_init, ✅Workflow:OpenAI_Compatible_Serving | Client, API, Integration |
| vllm-project_vllm_Chat_Completion_API | principles/vllm-project_vllm_Chat_Completion_API.md | ✅Impl:chat_completions_create, ✅Workflow:OpenAI_Compatible_Serving | NLP, Chat, API |
| vllm-project_vllm_Response_Handling | principles/vllm-project_vllm_Response_Handling.md | ✅Impl:ChatCompletion_processing, ✅Workflow:OpenAI_Compatible_Serving | NLP, Output_Processing |

### Multi_LoRA_Inference

| Page | File | Connections | Domains |
|------|------|-------------|---------|
| vllm-project_vllm_LoRA_Engine_Configuration | principles/vllm-project_vllm_LoRA_Engine_Configuration.md | ✅Impl:EngineArgs_lora, ✅Workflow:Multi_LoRA_Inference | LoRA, Configuration, Inference |
| vllm-project_vllm_LoRA_Adapter_Registration | principles/vllm-project_vllm_LoRA_Adapter_Registration.md | ✅Impl:LoRARequest_init, ✅Workflow:Multi_LoRA_Inference | LoRA, Adapters |
| vllm-project_vllm_LoRA_Request_Submission | principles/vllm-project_vllm_LoRA_Request_Submission.md | ✅Impl:LLMEngine_add_request_lora, ✅Workflow:Multi_LoRA_Inference | LoRA, Inference |
| vllm-project_vllm_LoRA_Scheduling | principles/vllm-project_vllm_LoRA_Scheduling.md | ✅Impl:Scheduler_lora_batching, ✅Workflow:Multi_LoRA_Inference | LoRA, Scheduling, Batching |
| vllm-project_vllm_LoRA_Output_Processing | principles/vllm-project_vllm_LoRA_Output_Processing.md | ✅Impl:RequestOutput_lora, ✅Workflow:Multi_LoRA_Inference | LoRA, Output_Processing |

### Vision_Language_Inference

| Page | File | Connections | Domains |
|------|------|-------------|---------|
| vllm-project_vllm_VLM_Model_Configuration | principles/vllm-project_vllm_VLM_Model_Configuration.md | ✅Impl:EngineArgs_vlm, ✅Workflow:Vision_Language_Inference | Vision, Configuration, Multimodal |
| vllm-project_vllm_Image_Input_Preparation | principles/vllm-project_vllm_Image_Input_Preparation.md | ✅Impl:MultiModalData_image, ✅Workflow:Vision_Language_Inference | Vision, Input_Processing |
| vllm-project_vllm_VLM_Prompt_Construction | principles/vllm-project_vllm_VLM_Prompt_Construction.md | ✅Impl:VLM_prompt_format, ✅Workflow:Vision_Language_Inference | Vision, NLP, Prompting |
| vllm-project_vllm_Multimodal_Generation | principles/vllm-project_vllm_Multimodal_Generation.md | ✅Impl:LLM_generate_mm, ✅Workflow:Vision_Language_Inference | Vision, NLP, Inference |
| vllm-project_vllm_VLM_Output_Processing | principles/vllm-project_vllm_VLM_Output_Processing.md | ✅Impl:RequestOutput_vlm, ✅Workflow:Vision_Language_Inference | Vision, Output_Processing |

### Speculative_Decoding

| Page | File | Connections | Domains |
|------|------|-------------|---------|
| vllm-project_vllm_Speculative_Method_Selection | principles/vllm-project_vllm_Speculative_Method_Selection.md | ✅Impl:SpeculativeMethod_choice, ✅Workflow:Speculative_Decoding | NLP, Optimization |
| vllm-project_vllm_Speculative_Configuration | principles/vllm-project_vllm_Speculative_Configuration.md | ✅Impl:SpeculativeConfig_init, ✅Workflow:Speculative_Decoding | NLP, Configuration |
| vllm-project_vllm_Speculative_Engine_Init | principles/vllm-project_vllm_Speculative_Engine_Init.md | ✅Impl:LLM_speculative, ✅Workflow:Speculative_Decoding | NLP, Inference, Systems |
| vllm-project_vllm_Speculative_Generation | principles/vllm-project_vllm_Speculative_Generation.md | ✅Impl:LLM_generate_spec, ✅Workflow:Speculative_Decoding | NLP, Inference, Optimization |
| vllm-project_vllm_Speculation_Metrics | principles/vllm-project_vllm_Speculation_Metrics.md | ✅Impl:get_metrics_spec, ✅Workflow:Speculative_Decoding | NLP, Monitoring, Optimization |

### Structured_Output_Generation

| Page | File | Connections | Domains |
|------|------|-------------|---------|
| vllm-project_vllm_Constraint_Definition | principles/vllm-project_vllm_Constraint_Definition.md | ✅Impl:StructuredOutputsParams_types, ✅Workflow:Structured_Output_Generation | NLP, Structured_Output, Constraints |
| vllm-project_vllm_StructuredOutputsParams_Configuration | principles/vllm-project_vllm_StructuredOutputsParams_Configuration.md | ✅Impl:StructuredOutputsParams_init, ✅Workflow:Structured_Output_Generation | NLP, Structured_Output, Configuration |
| vllm-project_vllm_Structured_SamplingParams | principles/vllm-project_vllm_Structured_SamplingParams.md | ✅Impl:SamplingParams_structured, ✅Workflow:Structured_Output_Generation | NLP, Structured_Output, Sampling |
| vllm-project_vllm_Constrained_Generation | principles/vllm-project_vllm_Constrained_Generation.md | ✅Impl:LLM_generate_structured, ✅Workflow:Structured_Output_Generation | NLP, Structured_Output, Inference |
| vllm-project_vllm_Structured_Output_Parsing | principles/vllm-project_vllm_Structured_Output_Parsing.md | ✅Impl:structured_output_parse, ✅Workflow:Structured_Output_Generation | NLP, Structured_Output, Output_Processing |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
