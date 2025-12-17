# Principle Index: vllm-project_vllm

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Summary

| Workflow | Principles | Status |
|----------|------------|--------|
| Basic_Offline_LLM_Inference | 6 | ✅ Complete |
| Online_API_Serving | 6 | ✅ Complete |
| Vision_Language_Multimodal_Inference | 6 | ✅ Complete |
| LoRA_Adapter_Inference | 6 | ✅ Complete |
| Speculative_Decoding | 5 | ✅ Complete |
| Distributed_Data_Parallel_Inference | 6 | ✅ Complete |
| **Total** | **35** | **100%** |

---

## Pages

### Basic_Offline_LLM_Inference Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_Engine_Configuration | [→](./principles/vllm-project_vllm_Engine_Configuration.md) | ✅Impl:vllm-project_vllm_EngineArgs | Engine parameter configuration |
| vllm-project_vllm_Sampling_Configuration | [→](./principles/vllm-project_vllm_Sampling_Configuration.md) | ✅Impl:vllm-project_vllm_SamplingParams | Sampling parameters |
| vllm-project_vllm_Model_Loading | [→](./principles/vllm-project_vllm_Model_Loading.md) | ✅Impl:vllm-project_vllm_LLM_init | LLM instantiation |
| vllm-project_vllm_Input_Formatting | [→](./principles/vllm-project_vllm_Input_Formatting.md) | ✅Impl:vllm-project_vllm_PromptType | Prompt formatting |
| vllm-project_vllm_Batch_Generation | [→](./principles/vllm-project_vllm_Batch_Generation.md) | ✅Impl:vllm-project_vllm_LLM_generate | Batch inference |
| vllm-project_vllm_Output_Processing | [→](./principles/vllm-project_vllm_Output_Processing.md) | ✅Impl:vllm-project_vllm_RequestOutput | Output handling |

### Online_API_Serving Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_Server_Configuration | [→](./principles/vllm-project_vllm_Server_Configuration.md) | ✅Impl:vllm-project_vllm_vllm_serve | Server setup |
| vllm-project_vllm_Server_Startup | [→](./principles/vllm-project_vllm_Server_Startup.md) | ✅Impl:vllm-project_vllm_vllm_serve_startup | Server lifecycle |
| vllm-project_vllm_API_Client_Setup | [→](./principles/vllm-project_vllm_API_Client_Setup.md) | ✅Impl:vllm-project_vllm_OpenAI_Client | Client configuration |
| vllm-project_vllm_Chat_Formatting | [→](./principles/vllm-project_vllm_Chat_Formatting.md) | ✅Impl:vllm-project_vllm_chat_message_format | Message structure |
| vllm-project_vllm_API_Request_Processing | [→](./principles/vllm-project_vllm_API_Request_Processing.md) | ✅Impl:vllm-project_vllm_chat_completions_create | Request execution |
| vllm-project_vllm_Streaming_Response | [→](./principles/vllm-project_vllm_Streaming_Response.md) | ✅Impl:vllm-project_vllm_sse_streaming | SSE streaming |

### Vision_Language_Multimodal_Inference Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_VLM_Configuration_Principle | [→](./principles/vllm-project_vllm_VLM_Configuration_Principle.md) | ✅Impl:vllm-project_vllm_EngineArgs_Multimodal_API | VLM config |
| vllm-project_vllm_Multimodal_Input_Preparation_Principle | [→](./principles/vllm-project_vllm_Multimodal_Input_Preparation_Principle.md) | ✅Impl:vllm-project_vllm_Image_Loading_API | Image loading |
| vllm-project_vllm_Multimodal_Prompt_Formatting_Principle | [→](./principles/vllm-project_vllm_Multimodal_Prompt_Formatting_Principle.md) | ✅Impl:vllm-project_vllm_VLM_Prompt_Templates_Pattern | Prompt templates |
| vllm-project_vllm_VLM_Engine_Initialization_Principle | [→](./principles/vllm-project_vllm_VLM_Engine_Initialization_Principle.md) | ✅Impl:vllm-project_vllm_LLM_Multimodal_Initialization_API | VLM engine init |
| vllm-project_vllm_Multimodal_Generation_Principle | [→](./principles/vllm-project_vllm_Multimodal_Generation_Principle.md) | ✅Impl:vllm-project_vllm_LLM_Generate_Multimodal_API | VLM generation |
| vllm-project_vllm_VLM_Output_Processing_Principle | [→](./principles/vllm-project_vllm_VLM_Output_Processing_Principle.md) | ✅Impl:vllm-project_vllm_RequestOutput_VLM_API | VLM output |

### LoRA_Adapter_Inference Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_LoRA_Engine_Configuration | [→](./principles/vllm-project_vllm_LoRA_Engine_Configuration.md) | ✅Impl:vllm-project_vllm_EngineArgs_lora | LoRA engine config |
| vllm-project_vllm_LoRA_Base_Model_Loading | [→](./principles/vllm-project_vllm_LoRA_Base_Model_Loading.md) | ✅Impl:vllm-project_vllm_LLMEngine_from_engine_args | Base model loading |
| vllm-project_vllm_LoRA_Adapter_Loading | [→](./principles/vllm-project_vllm_LoRA_Adapter_Loading.md) | ✅Impl:vllm-project_vllm_snapshot_download_lora | Adapter download |
| vllm-project_vllm_LoRA_Request_Creation | [→](./principles/vllm-project_vllm_LoRA_Request_Creation.md) | ✅Impl:vllm-project_vllm_LoRARequest | Request creation |
| vllm-project_vllm_MultiLoRA_Inference | [→](./principles/vllm-project_vllm_MultiLoRA_Inference.md) | ✅Impl:vllm-project_vllm_LLMEngine_add_request | Multi-LoRA exec |
| vllm-project_vllm_LoRA_Output_Processing | [→](./principles/vllm-project_vllm_LoRA_Output_Processing.md) | ✅Impl:vllm-project_vllm_RequestOutput_lora | LoRA output |

### Speculative_Decoding Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_spec_method_selection | [→](./principles/vllm-project_vllm_spec_method_selection.md) | ✅Impl:vllm-project_vllm_SpeculativeConfig | Method selection |
| vllm-project_vllm_speculative_engine_init | [→](./principles/vllm-project_vllm_speculative_engine_init.md) | ✅Impl:vllm-project_vllm_LLM_speculative | Spec config and engine init |
| vllm-project_vllm_speculative_prompt_prep | [→](./principles/vllm-project_vllm_speculative_prompt_prep.md) | ✅Impl:vllm-project_vllm_TokensPrompt_spec | Prompt prep |
| vllm-project_vllm_speculative_generation | [→](./principles/vllm-project_vllm_speculative_generation.md) | ✅Impl:vllm-project_vllm_LLM_generate_spec | Spec generation |
| vllm-project_vllm_speculative_metrics | [→](./principles/vllm-project_vllm_speculative_metrics.md) | ✅Impl:vllm-project_vllm_get_metrics | Metrics analysis |

### Distributed_Data_Parallel_Inference Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_strategy_planning | [→](./principles/vllm-project_vllm_strategy_planning.md) | ✅Impl:vllm-project_vllm_ParallelConfig | Strategy planning |
| vllm-project_vllm_dp_env_vars | [→](./principles/vllm-project_vllm_dp_env_vars.md) | ✅Impl:vllm-project_vllm_process_launcher | Env setup |
| vllm-project_vllm_LLM_distributed | [→](./principles/vllm-project_vllm_LLM_distributed.md) | ✅Impl:vllm-project_vllm_LLM_class | Engine init |
| vllm-project_vllm_prompt_partitioning | [→](./principles/vllm-project_vllm_prompt_partitioning.md) | ✅Impl:vllm-project_vllm_data_partition_impl | Data partition |
| vllm-project_vllm_LLM_generate_dp | [→](./principles/vllm-project_vllm_LLM_generate_dp.md) | ✅Impl:vllm-project_vllm_generate_method | Parallel exec |
| vllm-project_vllm_result_aggregation | [→](./principles/vllm-project_vllm_result_aggregation.md) | ✅Impl:vllm-project_vllm_result_collector | Result aggregation |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
