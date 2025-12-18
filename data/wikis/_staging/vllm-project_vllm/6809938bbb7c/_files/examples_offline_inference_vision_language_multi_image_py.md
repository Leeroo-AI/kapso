# File: `examples/offline_inference/vision_language_multi_image.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 1542 |
| Classes | `ModelRequestData` |
| Functions | `load_aria`, `load_aya_vision`, `load_bee`, `load_command_a_vision`, `load_deepseek_vl2`, `load_deepseek_ocr`, `load_gemma3`, `load_h2ovl`, `... +35 more` |
| Imports | PIL, argparse, dataclasses, huggingface_hub, os, transformers, typing, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Specialized reference for vision-language models supporting multiple images per prompt (40+ models).

**Mechanism:** Provides model-specific loader functions for multi-image VLMs including Aria, Aya Vision, Bee, Command-A Vision, DeepSeek variants, Gemma3, GLM-4.5V, H2OVL, HunyuanOCR, HyperCLOVAX-SEED, Idefics3, InternVL, Keye-VL, Kimi-VL, Llama 4, LLaVA variants, Mistral3, NVLM-D, Ovis, Phi-3V/4, Pixtral, Qwen-VL, SmolVLM, Step3, Tarsier, and more. Each function handles proper image list formatting, limit_mm_per_prompt configuration, and multi-image placeholder syntax.

**Significance:** Critical reference for multi-image inference patterns. Shows how different VLM architectures handle multiple images with varying placeholder syntax, image ordering, and configuration requirements. Essential for applications processing image galleries, multi-page documents, or video frames.
