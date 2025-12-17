# Heuristic Index: vllm-project_vllm

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_GPU_Memory_Utilization | [→](./heuristics/vllm-project_vllm_GPU_Memory_Utilization.md) | ✅Impl:vllm-project_vllm_EngineArgs, ✅Impl:vllm-project_vllm_LLM_init, ✅Workflow:vllm-project_vllm_Basic_Offline_LLM_Inference | Memory tuning for KV cache optimization |
| vllm-project_vllm_Temperature_Sampling | [→](./heuristics/vllm-project_vllm_Temperature_Sampling.md) | ✅Impl:vllm-project_vllm_SamplingParams, ✅Workflow:vllm-project_vllm_Basic_Offline_LLM_Inference, ✅Workflow:vllm-project_vllm_Online_API_Serving | Temperature clamping to avoid NaN/Inf |
| vllm-project_vllm_Tensor_Parallelism | [→](./heuristics/vllm-project_vllm_Tensor_Parallelism.md) | ✅Impl:vllm-project_vllm_EngineArgs, ✅Workflow:vllm-project_vllm_Online_API_Serving, ✅Workflow:vllm-project_vllm_Distributed_Data_Parallel_Inference | Multi-GPU tensor parallel configuration |
| vllm-project_vllm_Max_Model_Length | [→](./heuristics/vllm-project_vllm_Max_Model_Length.md) | ✅Impl:vllm-project_vllm_EngineArgs, ✅Impl:vllm-project_vllm_LLM_init, ✅Workflow:vllm-project_vllm_Basic_Offline_LLM_Inference | Context length tuning for memory management |
| vllm-project_vllm_Enforce_Eager_Mode | [→](./heuristics/vllm-project_vllm_Enforce_Eager_Mode.md) | ✅Impl:vllm-project_vllm_EngineArgs, ✅Workflow:vllm-project_vllm_Online_API_Serving | Debugging mode to disable CUDA graphs |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
