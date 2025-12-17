# Workflow Index: vllm-project_vllm

> Comprehensive index of Workflows and their implementation context.
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Implementation APIs |
|----------|-------|------------|---------------------|
| Basic_Offline_LLM_Inference | 6 | 6 | LLM, SamplingParams, EngineArgs |
| Online_API_Serving | 6 | 6 | OpenAI SDK, vllm serve, FastAPI |
| Vision_Language_Multimodal_Inference | 6 | 6 | LLM, EngineArgs, multimodal processors |
| LoRA_Adapter_Inference | 6 | 6 | LLMEngine, LoRARequest, enable_lora |
| Speculative_Decoding | 6 | 5 | LLM, speculative_config, EAGLE |
| Distributed_Data_Parallel_Inference | 6 | 6 | LLM, tensor_parallel_size, VLLM_DP_* |

---

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./workflows/vllm-project_vllm_Basic_Offline_LLM_Inference.md) | ✅Principle:vllm-project_vllm_Engine_Configuration, ✅Principle:vllm-project_vllm_Sampling_Configuration, ✅Principle:vllm-project_vllm_Model_Loading, ✅Principle:vllm-project_vllm_Input_Formatting, ✅Principle:vllm-project_vllm_Batch_Generation, ✅Principle:vllm-project_vllm_Output_Processing | Core offline inference workflow |
| vllm-project_vllm_Online_API_Serving | [→](./workflows/vllm-project_vllm_Online_API_Serving.md) | ✅Principle:vllm-project_vllm_Server_Configuration, ✅Principle:vllm-project_vllm_Server_Startup, ✅Principle:vllm-project_vllm_API_Client_Setup, ✅Principle:vllm-project_vllm_Chat_Formatting, ✅Principle:vllm-project_vllm_API_Request_Processing, ✅Principle:vllm-project_vllm_Streaming_Response | OpenAI-compatible API |
| vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./workflows/vllm-project_vllm_Vision_Language_Multimodal_Inference.md) | ✅Principle:vllm-project_vllm_VLM_Configuration_Principle, ✅Principle:vllm-project_vllm_Multimodal_Input_Preparation_Principle, ✅Principle:vllm-project_vllm_Multimodal_Prompt_Formatting_Principle, ✅Principle:vllm-project_vllm_VLM_Engine_Initialization_Principle, ✅Principle:vllm-project_vllm_Multimodal_Generation_Principle, ✅Principle:vllm-project_vllm_VLM_Output_Processing_Principle | 60+ VLM models |
| vllm-project_vllm_LoRA_Adapter_Inference | [→](./workflows/vllm-project_vllm_LoRA_Adapter_Inference.md) | ✅Principle:vllm-project_vllm_LoRA_Engine_Configuration, ✅Principle:vllm-project_vllm_LoRA_Base_Model_Loading, ✅Principle:vllm-project_vllm_LoRA_Adapter_Loading, ✅Principle:vllm-project_vllm_LoRA_Request_Creation, ✅Principle:vllm-project_vllm_MultiLoRA_Inference, ✅Principle:vllm-project_vllm_LoRA_Output_Processing | Multi-LoRA serving |
| vllm-project_vllm_Speculative_Decoding | [→](./workflows/vllm-project_vllm_Speculative_Decoding.md) | ✅Principle:vllm-project_vllm_spec_method_selection, ✅Principle:vllm-project_vllm_speculative_engine_init, ✅Principle:vllm-project_vllm_speculative_prompt_prep, ✅Principle:vllm-project_vllm_speculative_generation, ✅Principle:vllm-project_vllm_speculative_metrics | EAGLE/n-gram/MLP |
| vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./workflows/vllm-project_vllm_Distributed_Data_Parallel_Inference.md) | ✅Principle:vllm-project_vllm_strategy_planning, ✅Principle:vllm-project_vllm_dp_env_vars, ✅Principle:vllm-project_vllm_LLM_distributed, ✅Principle:vllm-project_vllm_prompt_partitioning, ✅Principle:vllm-project_vllm_LLM_generate_dp, ✅Principle:vllm-project_vllm_result_aggregation | TP/DP/EP parallelism |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
