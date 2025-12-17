# File: `examples/offline_inference/vision_language.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 2243 |
| Classes | `ModelRequestData` |
| Functions | `run_aria`, `run_aya_vision`, `run_bee`, `run_bagel`, `run_blip2`, `run_chameleon`, `run_command_a_vision`, `run_deepseek_vl2`, `... +60 more` |
| Imports | contextlib, dataclasses, huggingface_hub, os, random, transformers, typing, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive vision-language model examples (60+ models)

**Mechanism:** Massive example covering 60+ VLM models including Aria, BLIP-2, Chameleon, DeepSeek-VL2, Gemma3, GLM-4V, Idefics3, InternVL, LLaVA variants, Phi-3V, Qwen-VL families, etc. Each model has dedicated function with prompt format, engine config, mm_processor_kwargs, and stop tokens. Supports both image and video inputs with optional MM cache testing.

**Significance:** Comprehensive reference for VLM inference covering most vision-language models supported by vLLM.
