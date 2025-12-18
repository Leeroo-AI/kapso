# Phase 3: Enrichment Report

**Repository:** vllm-project_vllm
**Date:** 2025-01-15
**Phase:** Enrichment (Environments & Heuristics)

---

## Summary

Phase 3 mined the vLLM source code for environment constraints and tribal knowledge (heuristics). Created 4 Environment pages and 3 Heuristic pages based on analysis of:
- Implementation Index requirements (30 implementations)
- Source code patterns (`vllm/envs.py`, `vllm/entrypoints/llm.py`, `vllm/sampling_params.py`)
- Default values and configuration parameters

---

## Environments Created

| Environment | Required By (Count) | Notes |
|-------------|---------------------|-------|
| vllm-project_vllm_GPU_Environment | 11 implementations | Linux + CUDA 11.8+ or ROCm for GPU-accelerated LLM inference |
| vllm-project_vllm_Python_Environment | 14 implementations | Python 3.9+ with transformers, msgspec, pydantic for CPU-side operations |
| vllm-project_vllm_Server_Environment | 2 implementations | Linux server with uvicorn, FastAPI for HTTP API deployment |
| vllm-project_vllm_Client_Environment | 3 implementations | Python with OpenAI client library for API consumption |

### Environment Details

**GPU_Environment:**
- System: Linux (Ubuntu 20.04/22.04), NVIDIA GPU compute capability >= 7.0
- CUDA: 11.8 or 12.x, cuDNN 8.6+, NCCL 2.18+
- Python: torch >= 2.0.0, vllm, transformers >= 4.40.0, triton >= 2.2.0
- Credentials: HF_TOKEN (optional, for gated models)

**Python_Environment:**
- System: Linux/macOS/Windows, Python 3.9-3.12
- Python: transformers, tokenizers, msgspec, pydantic, pillow, numpy
- Used for: SamplingParams, PromptType, LoRARequest, StructuredOutputsParams

**Server_Environment:**
- Extends GPU_Environment
- Additional: uvicorn, fastapi, uvloop, httptools
- Credentials: VLLM_API_KEY (optional authentication)

**Client_Environment:**
- System: Any platform
- Python: openai >= 1.0.0, httpx
- Used for: OpenAI-compatible API consumption

---

## Heuristics Created

| Heuristic | Applies To | Notes |
|-----------|------------|-------|
| vllm-project_vllm_GPU_Memory_Utilization_Tuning | LLM_init, Basic_Offline_Inference | Tune gpu_memory_utilization (default 0.9) to balance KV cache vs OOM risk |
| vllm-project_vllm_Tensor_Parallel_Configuration | LLM_init, Basic_Offline_Inference, Online_Serving | Multi-GPU tensor parallelism setup for large models |
| vllm-project_vllm_Sampling_Temperature_Selection | SamplingParams_init, Basic_Offline_Inference, Online_Serving | Select temperature (0=greedy, 0.7=balanced, 1.2+=creative) |

### Heuristic Details

**GPU_Memory_Utilization_Tuning:**
- Default: 0.9 (90%)
- OOM fix: Reduce to 0.7-0.8
- Shared GPU: Use 0.5-0.7
- Maximum throughput: 0.9-0.95
- Source: `vllm/entrypoints/llm.py:208`

**Tensor_Parallel_Configuration:**
- Default: 1 (single GPU)
- 7B-13B models: TP=1
- 34B models: TP=2
- 70B models: TP=4
- 180B+ models: TP=8
- Trade-off: More GPUs = more memory but higher communication overhead
- Source: `vllm/entrypoints/llm.py:118-119`

**Sampling_Temperature_Selection:**
- Default: 1.0
- Greedy (deterministic): 0.0
- Balanced: 0.5-0.7
- Creative: 1.2+
- Source: `vllm/sampling_params.py:145-148`

---

## Links Added

### Environment Links Added
- 30 Environment links added to Implementation Index (all `⬜Env:` → `✅Env:`)

### Heuristic Connections
- GPU_Memory_Utilization_Tuning: 3 connections (1 Implementation, 1 Principle, 1 Workflow)
- Tensor_Parallel_Configuration: 3 connections (1 Implementation, 2 Workflows)
- Sampling_Temperature_Selection: 4 connections (1 Implementation, 1 Principle, 2 Workflows)

---

## Index Updates

| Index | Changes |
|-------|---------|
| _EnvironmentIndex.md | Added 4 Environment entries with full connection lists |
| _HeuristicIndex.md | Added 3 Heuristic entries with connection lists |
| _ImplementationIndex.md | Changed 30 `⬜Env:` references to `✅Env:vllm-project_vllm_*` |

---

## Notes for Audit Phase

### Potential Issues
1. **Implementation page links:** The Environment and Heuristic pages reference Implementation pages that may need reciprocal `[[requires_env::...]]` and `[[uses_heuristic::...]]` links added to complete bi-directional connections.

2. **Workflow references:** Heuristics reference workflows like `Basic_Offline_Inference` and `Online_Serving` - verify these workflow pages exist and add reciprocal links.

3. **Principle references:** Some Heuristics reference Principles (e.g., `LLM_Class_Initialization`, `SamplingParams_Configuration`) - verify these exist.

### Code Evidence Quality
- All Environment pages include Code Evidence section with actual source code snippets
- All Heuristic pages include Code Evidence with line numbers from source files
- Common Errors tables populated from known vLLM issues

### Coverage Assessment
- **GPU requirements:** Comprehensive coverage of CUDA/ROCm requirements
- **Python requirements:** Core dependencies documented; optional deps noted
- **Server/Client:** OpenAI-compatible API fully documented
- **Heuristics:** Key tuning parameters (memory, parallelism, sampling) covered

---

## Files Created

### Environments (4 files)
- `environments/vllm-project_vllm_GPU_Environment.md`
- `environments/vllm-project_vllm_Python_Environment.md`
- `environments/vllm-project_vllm_Server_Environment.md`
- `environments/vllm-project_vllm_Client_Environment.md`

### Heuristics (3 files)
- `heuristics/vllm-project_vllm_GPU_Memory_Utilization_Tuning.md`
- `heuristics/vllm-project_vllm_Tensor_Parallel_Configuration.md`
- `heuristics/vllm-project_vllm_Sampling_Temperature_Selection.md`

---

**Phase 3 Complete.**
