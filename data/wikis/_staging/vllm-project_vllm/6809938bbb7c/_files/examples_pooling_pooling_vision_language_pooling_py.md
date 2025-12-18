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

**Purpose:** Comprehensive multimodal embedding examples for vision-language models

**Mechanism:** Provides implementations for 6 different multimodal models (CLIP, SiGLIP, E5-V, VLM2Vec variants, JinaVL-Reranker) supporting text-only, image-only, text+image, and text+images modalities. Each model has custom prompt templates and configurations. Handles LoRA merging for VLM2Vec-Qwen2VL, supports both embedding and scoring tasks, and demonstrates proper multimodal data formatting for each architecture.

**Significance:** Comprehensive reference for vision-language pooling in vLLM, showcasing the diversity of multimodal embedding approaches. Critical for developers implementing image-text retrieval, visual question answering embeddings, and multimodal reranking. Demonstrates important patterns like handling image tokens, configuring MM processors, and merging LoRA adapters for deployment.
