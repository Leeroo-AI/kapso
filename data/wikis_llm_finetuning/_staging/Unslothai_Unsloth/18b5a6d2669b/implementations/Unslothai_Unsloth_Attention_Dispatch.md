# Implementation: Attention_Dispatch

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Attention]], [[domain::Infrastructure]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Unified attention backend selection and execution with support for FlashAttention, xFormers, and PyTorch SDPA.

=== Description ===
This module provides a unified interface for running attention operations across different backends. It automatically selects the best available backend (FlashAttention > xFormers > SDPA) and handles packed sequences, grouped query attention, and sliding window attention transparently.

=== Usage ===
Used by all model attention forward passes to dispatch to the optimal attention implementation based on available libraries and input characteristics.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/utils/attention_dispatch.py unsloth/utils/attention_dispatch.py]
* '''Lines:''' 1-275

=== Key Classes ===
<syntaxhighlight lang="python">
@dataclass
class AttentionConfig:
    """Per-layer attention metadata."""
    backend: str                    # Selected backend
    n_kv_heads: int                # Number of KV heads
    n_groups: int                  # GQA groups
    flash_dense_kwargs: dict       # FlashAttention dense kwargs
    flash_varlen_kwargs: dict      # FlashAttention varlen kwargs
    sdpa_kwargs: dict              # SDPA kwargs
    xformers_kwargs: dict          # xFormers kwargs

@dataclass
class AttentionContext:
    """Per-call info required to run attention."""
    bsz: int                       # Batch size
    q_len: int                     # Query sequence length
    kv_seq_len: int               # KV sequence length
    n_heads: int                   # Number of attention heads
    head_dim: int                  # Head dimension
    requires_grad: bool            # Training vs inference
    seq_info: Optional[Tuple]      # Packed sequence info
    attention_mask: Optional[Tensor]
    causal_mask: Optional[Any]
    sliding_window: Optional[int]  # Sliding window size
</syntaxhighlight>

=== Key Functions ===
<syntaxhighlight lang="python">
def select_attention_backend(use_varlen: bool = False) -> str:
    """
    Return attention backend based on availability.

    Priority: flash_varlen > flash_dense > xformers > sdpa

    Args:
        use_varlen: Prefer varlen flash for packed sequences

    Returns:
        Backend name: "flash_varlen", "flash_dense", "xformers", or "sdpa"
    """

def run_attention(
    *,
    config: AttentionConfig,
    context: AttentionContext,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
) -> Tensor:
    """
    Run attention using config/context info.

    Handles GQA expansion, packed sequence masking, and
    sliding window attention for each backend.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.utils.attention_dispatch import (
    AttentionConfig,
    AttentionContext,
    select_attention_backend,
    run_attention,
    FLASH_VARLEN,
    FLASH_DENSE,
    XFORMERS,
    SDPA,
)
</syntaxhighlight>

== Backend Priority ==

{| class="wikitable"
|-
! Priority !! Backend !! Constant !! Condition
|-
| 1 || FlashAttention varlen || FLASH_VARLEN || HAS_FLASH_ATTENTION and seq_info present
|-
| 2 || FlashAttention dense || FLASH_DENSE || HAS_FLASH_ATTENTION
|-
| 3 || xFormers || XFORMERS || HAS_XFORMERS
|-
| 4 || PyTorch SDPA || SDPA || Always available (fallback)
|}

== Usage Examples ==

=== Select Backend ===
<syntaxhighlight lang="python">
from unsloth.utils.attention_dispatch import select_attention_backend

# For packed sequences (sample packing)
backend = select_attention_backend(use_varlen=True)
# Returns "flash_varlen" if FlashAttention installed

# For standard batches
backend = select_attention_backend(use_varlen=False)
# Returns "flash_dense" or "xformers" or "sdpa"
</syntaxhighlight>

=== Run Attention ===
<syntaxhighlight lang="python">
from unsloth.utils.attention_dispatch import (
    AttentionConfig, AttentionContext, run_attention
)

config = AttentionConfig(
    backend="flash_dense",
    n_kv_heads=8,
    n_groups=4,  # 32 Q heads / 8 KV heads
    flash_dense_kwargs={"causal": True},
)

context = AttentionContext(
    bsz=2,
    q_len=512,
    kv_seq_len=512,
    n_heads=32,
    head_dim=128,
    requires_grad=True,
    seq_info=None,
    attention_mask=None,
    causal_mask=None,
)

output = run_attention(config=config, context=context, Q=Q, K=K, V=V)
# output shape: [bsz, q_len, n_heads, head_dim]
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
