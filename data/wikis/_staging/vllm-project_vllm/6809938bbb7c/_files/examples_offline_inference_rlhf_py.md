# File: `examples/offline_inference/rlhf.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 147 |
| Classes | `MyLLM` |
| Imports | os, ray, rlhf_utils, torch, transformers, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates RLHF training pattern using vLLM for generation and separate training processes.

**Mechanism:** Creates custom MyLLM class that wraps vLLM's LLM with WorkerExtension for exposing model parameters. Generates responses using vLLM, computes rewards, and performs parameter updates. Uses Ray for distributed execution with separate actor processes for policy and reference models.

**Significance:** Shows integration pattern for RLHF (Reinforcement Learning from Human Feedback) workflows where vLLM handles efficient generation while PyTorch handles training. Important for fine-tuning with policy gradient methods like PPO.
