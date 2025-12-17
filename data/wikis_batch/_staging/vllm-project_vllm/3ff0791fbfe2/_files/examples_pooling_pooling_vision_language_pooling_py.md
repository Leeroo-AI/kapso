# File: `examples/pooling/pooling/vision_language_pooling.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 410 |
| Classes | `TextQuery`, `ImageQuery`, `TextImageQuery`, `TextImagesQuery`, `ModelRequestData` |
| Functions | `run_clip`, `run_e5_v`, `run_jinavl_reranker`, `run_siglip`, `run_vlm2vec_phi3v`, `run_vlm2vec_qwen2vl`, `get_query`, `run_encode`, `... +3 more` |
| Imports | PIL, argparse, dataclasses, pathlib, typing, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Vision-language multimodal pooling examples

**Mechanism:** Comprehensive demonstration of multimodal pooling with various vision-language models. Supports multiple modalities (text-only, image-only, text+image, text+images) and models (CLIP, SigLIP, E5-V, VLM2Vec variants, Jina reranker). Each model has a dedicated setup function that configures appropriate prompts, engine arguments, and handles model-specific requirements like LoRA merging for VLM2Vec-Qwen2VL. Provides both embedding and scoring task modes.

**Significance:** Advanced reference implementation showcasing vLLM's multimodal capabilities across diverse architectures. Demonstrates prompt formatting, multimodal data handling, and task-specific configuration for embedding extraction and document reranking with vision-language models.
