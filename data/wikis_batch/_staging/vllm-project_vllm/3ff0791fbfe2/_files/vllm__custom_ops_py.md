# File: `vllm/_custom_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3080 |
| Classes | `CPUDNNLGEMMHandler` |
| Functions | `paged_attention_v1`, `paged_attention_v2`, `paged_attention_rocm`, `mla_decode_kvcache_cpu`, `merge_attn_states`, `convert_vertical_slash_indexes`, `convert_vertical_slash_indexes_mergehead`, `rotary_embedding`, `... +119 more` |
| Imports | torch, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Custom PyTorch operations registry

**Mechanism:** Central registry of all custom CUDA/ROCm/CPU operations used throughout vLLM. Contains 100+ operation definitions including attention mechanisms (paged_attention variants), quantization operations (GPTQ, AWQ, FP8, MXFP4), activation functions, MoE operations, sampling kernels, and specialized matrix operations. Operations are either direct bindings to compiled extensions or Python fallback implementations. Includes platform-specific dispatch logic and operation factories.

**Significance:** Core performance infrastructure that enables vLLM's high-speed inference. These custom operations are highly optimized for specific hardware and quantization schemes, providing orders of magnitude speedup over standard PyTorch ops. Central to vLLM's ability to efficiently serve LLMs on various hardware platforms (CUDA, ROCm, CPU).
