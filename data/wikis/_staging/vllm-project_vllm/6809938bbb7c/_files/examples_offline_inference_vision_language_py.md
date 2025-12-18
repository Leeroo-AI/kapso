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

**Purpose:** Comprehensive reference covering 60+ vision-language models with model-specific configurations and prompt formats.

**Mechanism:** Provides dedicated configuration functions for each VLM family (Aria, Aya, Bee, BLIP-2, Chameleon, Command-A, DeepSeek-VL, Gemma3, GLM-4.5V, H2OVL, InternVL, Llama 4, LLaVA variants, Mistral3, NVLM, Ovis, Phi-3V, Phi-4, Pixtral, Qwen-VL variants, SmolVLM, Step3, and many more). Each function returns proper EngineArgs, image placeholder formatting, chat templates, and model-specific requirements.

**Significance:** Essential reference guide for vision-language inference in vLLM. Covers vast ecosystem of VLMs with correct prompt engineering, image token placement, tensor parallelism settings, and multimodal configuration for each architecture. Critical for anyone working with VLMs in vLLM.
