# Implementation: Flex_Attention

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|Gemma2|https://arxiv.org/abs/2408.00118]]
|-
! Domains
| [[domain::Kernels]], [[domain::Attention]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized attention kernel implementations supporting logit softcapping for Gemma2-style models using PyTorch's Flex Attention API.

=== Description ===
This module provides attention implementations with logit softcapping, a technique used by Gemma2 models to prevent attention logits from becoming too large. It provides both a fast path using PyTorch's `flex_attention` (torch 2.5+) and a fallback `torch.compile`-optimized implementation for older PyTorch versions. The softcapping applies tanh scaling: `t * tanh(logits / t)` before softmax.

=== Usage ===
Use these functions when working with Gemma2 or similar models that require logit softcapping in attention. The module automatically selects the best implementation based on PyTorch version.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/flex_attention.py unsloth/kernels/flex_attention.py]
* '''Lines:''' 1-187

=== Signature ===
<syntaxhighlight lang="python">
def slow_attention_softcapping(
    Q: torch.Tensor,       # [batch, n_heads, seq_len, head_dim]
    K: torch.Tensor,       # [batch, n_kv_heads, seq_len, head_dim]
    V: torch.Tensor,       # [batch, n_kv_heads, seq_len, head_dim]
    causal_mask: torch.Tensor,  # [seq_len, seq_len] or block_mask
    self,                  # Model layer (for config access)
    bsz: int,             # Batch size
    q_len: int,           # Query sequence length
) -> torch.Tensor:        # [batch, seq_len, hidden_dim]
    """
    Compute attention with logit softcapping.

    Uses config values:
        - self.config.query_pre_attn_scalar (s): Scaling factor
        - self.config.attn_logit_softcapping (t): Softcap threshold

    Applies: t * tanh(QK^T / (sqrt(s) * t)) before softmax.
    """

def slow_inference_attention_softcapping(...) -> torch.Tensor:
    """Non-compiled version for inference (no torch.compile overhead)."""

def create_flex_attention_causal_mask(max_seq_length: int = 8192):
    """Create causal mask for flex_attention (torch 2.5+)."""

def create_flex_attention_sliding_window_mask(
    max_seq_length: int = 8192,
    sliding_window: int = 4096,
):
    """Create sliding window causal mask for flex_attention."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.flex_attention import (
    slow_attention_softcapping,
    slow_inference_attention_softcapping,
    create_flex_attention_causal_mask,
    HAS_FLEX_ATTENTION,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Q || Tensor || Yes || Query tensor [batch, n_heads, seq_len, head_dim]
|-
| K || Tensor || Yes || Key tensor [batch, n_kv_heads, seq_len, head_dim]
|-
| V || Tensor || Yes || Value tensor [batch, n_kv_heads, seq_len, head_dim]
|-
| causal_mask || Tensor || Yes || Causal attention mask
|-
| self || nn.Module || Yes || Layer module with config for softcap params
|-
| bsz || int || Yes || Batch size
|-
| q_len || int || Yes || Query sequence length
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || Tensor || Attention output [batch, seq_len, n_heads * head_dim]
|}

== Usage Examples ==

=== Using Softcapped Attention ===
<syntaxhighlight lang="python">
from unsloth.kernels.flex_attention import slow_attention_softcapping

# Inside a Gemma2-style attention forward pass
# Q, K, V shapes: [batch, n_heads, seq_len, head_dim]
output = slow_attention_softcapping(
    Q=query_states,
    K=key_states,
    V=value_states,
    causal_mask=attention_mask,
    self=self,  # Has config.query_pre_attn_scalar and config.attn_logit_softcapping
    bsz=batch_size,
    q_len=seq_length,
)
# output shape: [batch, seq_len, hidden_size]
</syntaxhighlight>

=== Check Flex Attention Availability ===
<syntaxhighlight lang="python">
from unsloth.kernels.flex_attention import HAS_FLEX_ATTENTION

if HAS_FLEX_ATTENTION:
    print("Using PyTorch 2.5+ flex_attention (faster)")
else:
    print("Using torch.compile fallback")
</syntaxhighlight>

=== Create Block Masks ===
<syntaxhighlight lang="python">
from unsloth.kernels.flex_attention import (
    create_flex_attention_causal_mask,
    create_flex_attention_sliding_window_mask,
)

# Standard causal mask
causal_mask = create_flex_attention_causal_mask(max_seq_length=8192)

# Sliding window mask (e.g., for Mistral-style attention)
sliding_mask = create_flex_attention_sliding_window_mask(
    max_seq_length=8192,
    sliding_window=4096,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
