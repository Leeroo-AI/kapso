# File: `src/peft/tuners/adaption_prompt/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 236 |
| Classes | `_BaseAdaptedAttention`, `AdaptedAttentionGPT`, `AdaptedAttention` |
| Imports | config, math, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Attention layer wrappers that inject learnable adaption prompts

**Mechanism:** _BaseAdaptedAttention wraps attention modules with adaption_prompt (learnable embeddings) and adaption_gate (zero-initialized scaling). AdaptedAttention (LLaMA/Mistral) and AdaptedAttentionGPT compute keys/values from prompts, calculate attention scores against query states, apply gated softmax, and add prompt output to base attention: output = base_output + gate * softmax(Q @ K_prompt.T) @ V_prompt

**Significance:** Implements zero-init gating mechanism from LLaMA-Adapter paper - allows gradual incorporation of learned prompts without disrupting pretrained attention patterns, enabling stable fine-tuning with minimal trainable parameters
