# Heuristic Index: vllm-project_vllm

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| vllm-project_vllm_GPU_Memory_Utilization_Tuning | [→](./heuristics/vllm-project_vllm_GPU_Memory_Utilization_Tuning.md) | ✅Impl:vllm-project_vllm_LLM_init, ✅Principle:vllm-project_vllm_LLM_Class_Initialization, ✅Workflow:vllm-project_vllm_Basic_Offline_Inference | Tune gpu_memory_utilization (default 0.9) to balance KV cache vs OOM risk |
| vllm-project_vllm_Tensor_Parallel_Configuration | [→](./heuristics/vllm-project_vllm_Tensor_Parallel_Configuration.md) | ✅Impl:vllm-project_vllm_LLM_init, ✅Workflow:vllm-project_vllm_Basic_Offline_Inference, ✅Workflow:vllm-project_vllm_OpenAI_Compatible_Serving, ✅Principle:vllm-project_vllm_LLM_Class_Initialization | Multi-GPU tensor parallelism setup for large models |
| vllm-project_vllm_Sampling_Temperature_Selection | [→](./heuristics/vllm-project_vllm_Sampling_Temperature_Selection.md) | ✅Impl:vllm-project_vllm_SamplingParams_init, ✅Workflow:vllm-project_vllm_Basic_Offline_Inference, ✅Workflow:vllm-project_vllm_OpenAI_Compatible_Serving, ✅Principle:vllm-project_vllm_Sampling_Parameters | Select temperature (0=greedy, 0.7=balanced, 1.2+=creative) |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
