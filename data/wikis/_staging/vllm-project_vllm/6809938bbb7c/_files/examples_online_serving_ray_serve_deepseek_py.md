# File: `examples/online_serving/ray_serve_deepseek.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 55 |
| Imports | ray |

## Understanding

**Status:** ✅ Explored

**Purpose:** Ray Serve deployment configuration for DeepSeek models

**Mechanism:** Configures Ray Serve LLM with specific settings for DeepSeek R1/V3 models including multi-GPU parallelism (TP=8, PP=2), memory optimization (0.92 utilization), and production features (chunked prefill, prefix caching). Uses build_openai_app to create OpenAI-compatible endpoint. Specifies H100 accelerator type and autoscaling configuration.

**Significance:** Production deployment template for large-scale DeepSeek models. Demonstrates Ray Serve's integration with vLLM engine for scalable model serving. Important for enterprise deployments requiring automatic scaling, load balancing, and multi-node orchestration. Shows best practices for configuring large reasoning models (DeepSeek) with appropriate parallelism strategies and memory settings.
