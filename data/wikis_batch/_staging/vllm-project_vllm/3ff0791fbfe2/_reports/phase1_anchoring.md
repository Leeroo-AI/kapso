# Phase 1: Anchoring Report

## Summary
- Workflows created: 6
- Total steps documented: 36
- Implementation hints captured: 36

## Workflows Created

| Workflow | Source Files | Steps | Implementation APIs |
|----------|--------------|-------|---------------------|
| Basic_Offline_LLM_Inference | `vllm/__init__.py`, `examples/offline_inference/*.py` | 6 | LLM, SamplingParams, EngineArgs, RequestOutput |
| Online_API_Serving | `examples/online_serving/*.py` | 6 | vllm serve, OpenAI SDK, FastAPI |
| Vision_Language_Multimodal_Inference | `examples/offline_inference/vision_language.py`, `examples/offline_inference/audio_language.py` | 6 | LLM, EngineArgs (mm settings), fetch_image |
| LoRA_Adapter_Inference | `examples/offline_inference/multilora_inference.py`, `examples/offline_inference/lora_with_quantization_inference.py` | 6 | LLMEngine, LoRARequest, enable_lora |
| Speculative_Decoding | `examples/offline_inference/spec_decode.py`, `examples/offline_inference/mlpspeculator.py` | 6 | LLM, speculative_config, EAGLE |
| Distributed_Data_Parallel_Inference | `examples/offline_inference/data_parallel.py`, `examples/offline_inference/torchrun_*.py` | 6 | LLM, tensor_parallel_size, VLLM_DP_* env vars |

## Coverage Summary
- Source files covered: 55+
- Example files documented: 50+ (out of 77 example files)
- Core vllm files covered: 7 (sampling_params.py, outputs.py, __init__.py, envs.py, etc.)

## Implementation Context Captured

| Workflow | Principles | API Docs | Wrapper Docs | Pattern Docs |
|----------|------------|----------|--------------|--------------|
| Basic_Offline_LLM_Inference | 6 | 6 | 0 | 0 |
| Online_API_Serving | 6 | 2 | 2 | 2 |
| Vision_Language_Multimodal_Inference | 6 | 5 | 0 | 1 |
| LoRA_Adapter_Inference | 6 | 5 | 1 | 0 |
| Speculative_Decoding | 6 | 5 | 0 | 1 |
| Distributed_Data_Parallel_Inference | 6 | 4 | 0 | 2 |

## Notes for Excavation Phase

### APIs to Extract (with Source Locations)

| API | Source | Used By Principles |
|-----|--------|-------------------|
| `LLM` | `vllm/entrypoints/llm.py` | Model_Loading, VLM_Engine_Initialization, Speculative_Engine_Initialization, Distributed_Engine_Initialization |
| `LLM.generate` | `vllm/entrypoints/llm.py` | Batch_Generation, Multimodal_Generation, Speculative_Generation, Parallel_Inference_Execution |
| `LLM.get_metrics` | `vllm/entrypoints/llm.py` | Speculative_Metrics_Analysis |
| `SamplingParams` | `vllm/sampling_params.py:L111-597` | Sampling_Configuration |
| `EngineArgs` | `vllm/engine/arg_utils.py` | Engine_Configuration, VLM_Configuration, LoRA_Engine_Configuration |
| `LLMEngine` | `vllm/engine/llm_engine.py` | LoRA_Base_Model_Loading |
| `LLMEngine.from_engine_args` | `vllm/engine/llm_engine.py` | LoRA_Base_Model_Loading |
| `LLMEngine.add_request` | `vllm/engine/llm_engine.py` | MultiLoRA_Inference |
| `RequestOutput` | `vllm/outputs.py:L1-345` | Output_Processing, VLM_Output_Processing, LoRA_Output_Processing |
| `LoRARequest` | `vllm/lora/request.py` | LoRA_Request_Creation |
| `TextPrompt`, `TokensPrompt` | `vllm/inputs.py` | Input_Formatting, Speculative_Prompt_Preparation |
| `fetch_image` | `vllm/multimodal/utils.py` | Multimodal_Input_Preparation |

### External Dependencies to Document

| Library | Used By | Purpose |
|---------|---------|---------|
| `openai` (Python SDK) | Online_API_Serving | OpenAI-compatible client |
| `transformers` | All workflows | Model loading, tokenization |
| `huggingface_hub` | LoRA_Adapter_Inference | `snapshot_download` for adapters |
| `fastapi` | Online_API_Serving | HTTP server framework |
| `uvicorn` | Online_API_Serving | ASGI server |
| `peft` | LoRA_Adapter_Inference | LoRA adapter support |
| `torch` | All workflows | Tensor operations, CUDA |
| `PIL` | Vision_Language_Multimodal_Inference | Image handling |

### User-Defined Patterns to Document

| Pattern | Workflow | Description |
|---------|----------|-------------|
| Chat message formatting | Online_API_Serving | Structure of `{"role": "...", "content": "..."}` messages |
| VLM prompt templates | Vision_Language_Multimodal_Inference | Model-specific image placeholder tokens (`<image>`, `<|image|>`, etc.) |
| Data partitioning | Distributed_Data_Parallel_Inference | Splitting prompts across DP ranks |
| Result aggregation | Distributed_Data_Parallel_Inference | Collecting outputs from distributed workers |
| Speculative method selection | Speculative_Decoding | Choosing between EAGLE/n-gram/MLP methods |

### Environment Variables to Document

| Variable | Source | Purpose |
|----------|--------|---------|
| `VLLM_DP_RANK` | `vllm/envs.py` | Data parallel rank index |
| `VLLM_DP_SIZE` | `vllm/envs.py` | Total data parallel replicas |
| `VLLM_DP_MASTER_IP` | `vllm/envs.py` | Coordination server IP |
| `VLLM_DP_MASTER_PORT` | `vllm/envs.py` | Coordination server port |

## Key Findings

### Repository Structure Insights

1. **Two Main Entry Points**:
   - `LLM` class for offline batch inference
   - `vllm serve` CLI for online API serving

2. **PagedAttention as Core Innovation**:
   - All workflows benefit from efficient KV cache management
   - Documented in the original vLLM paper (arxiv:2309.06180)

3. **Multi-Backend Support**:
   - NVIDIA CUDA (primary)
   - AMD ROCm
   - Intel CPU/GPU (IPEX)
   - TPU support

4. **Extensive Example Coverage**:
   - 34 offline inference examples
   - 27 online serving examples
   - 16 pooling/embedding examples

### Workflow Interactions

```
Basic_Offline_LLM_Inference (foundation)
    ├── Vision_Language_Multimodal_Inference (extends with images)
    ├── LoRA_Adapter_Inference (extends with adapters)
    ├── Speculative_Decoding (extends with draft models)
    └── Distributed_Data_Parallel_Inference (extends with parallelism)

Online_API_Serving (alternative entry point)
    └── Uses same underlying engine as Basic_Offline_LLM_Inference
```

### Quantization Integration

Multiple workflows support quantization:
- `LoRA_Adapter_Inference`: QLoRA, AWQ, GPTQ with LoRA
- `Basic_Offline_LLM_Inference`: FP8, INT8, INT4 via `quantization` parameter
- Not documented as separate workflow since it's a configuration option

## Files Not Covered

The following file categories were not mapped to workflows:
- Benchmark files (`benchmarks/`) - performance testing infrastructure
- Test files (`tests/`) - unit/integration tests
- Build tools (`setup.py`, `cmake/`) - compilation infrastructure
- RLHF examples - could be a future workflow

## Recommendations for Phase 2

1. **Prioritize Core APIs**:
   - `LLM` class is used by 4+ workflows
   - `SamplingParams` is fundamental to all generation

2. **Create Shared Principles**:
   - Some principles (Engine_Configuration, Model_Loading) may be shared across workflows
   - Consider creating base principles that specializations extend

3. **Document VLM Templates**:
   - The 60+ VLM model templates in `vision_language.py` should be documented
   - Each model has unique prompt formatting requirements

4. **Environment Documentation**:
   - vLLM has 200+ environment variables (`vllm/envs.py`)
   - Focus on variables used by documented workflows
