# File: `src/peft/tuners/adaption_prompt/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 236 |
| Classes | `_BaseAdaptedAttention`, `AdaptedAttentionGPT`, `AdaptedAttention` |
| Imports | config, math, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Adapted attention layer implementations

**Mechanism:** _BaseAdaptedAttention initializes adaption_prompt (learnable tokens) and adaption_gate (zero-init scaling). AdaptedAttentionGPT/AdaptedAttention wrap GPT2Attention/LlamaAttention, inject prompts as key/value pairs, compute attention scores (query @ adapter_k), apply gate, and add adapter_output to base output.

**Significance:** Core layer implementing LLaMA-Adapter attention mechanism. Trainable prompts with gated injection enable efficient adaptation while preserving base model weights.
