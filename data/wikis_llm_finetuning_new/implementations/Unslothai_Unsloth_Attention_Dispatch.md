# Implementation: Attention Dispatch

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::GPU_Optimization]], [[domain::Attention_Mechanisms]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

The `attention_dispatch.py` module provides a unified attention backend selection and execution system for Unsloth. It abstracts away the complexity of different attention implementations (FlashAttention, xFormers, PyTorch SDPA) and automatically selects the optimal backend based on hardware availability and input characteristics.

Key responsibilities:
* Define dataclasses for attention configuration and context
* Select optimal attention backend based on availability
* Execute attention with automatic backend dispatch
* Handle grouped query attention (GQA) and packed sequence inputs

== Code Reference ==

'''File:''' `unsloth/utils/attention_dispatch.py` (274 lines)

=== Backend Constants ===

<syntaxhighlight lang="python">
FLASH_VARLEN = "flash_varlen"
FLASH_DENSE = "flash_dense"
XFORMERS = "xformers"
SDPA = "sdpa"

HAS_FLASH_ATTENTION = ...  # imported from _utils
HAS_XFORMERS = xformers is not None
SDPA_HAS_GQA = "enable_gqa" in (scaled_dot_product_attention.__doc__ or "")
</syntaxhighlight>

=== AttentionConfig Dataclass ===

<syntaxhighlight lang="python">
@dataclass
class AttentionConfig:
    """Per-layer attention metadata."""
    backend: str
    n_kv_heads: int
    n_groups: int
    flash_dense_kwargs: Optional[dict[str, Any]] = None
    flash_varlen_kwargs: Optional[dict[str, Any]] = None
    sdpa_kwargs: Optional[dict[str, Any]] = None
    xformers_kwargs: Optional[dict[str, Any]] = None
</syntaxhighlight>

=== AttentionContext Dataclass ===

<syntaxhighlight lang="python">
@dataclass
class AttentionContext:
    """Per-call info required to run attention."""
    bsz: int
    q_len: int
    kv_seq_len: int
    n_heads: int
    head_dim: int
    requires_grad: bool
    seq_info: Optional[Tuple[Tensor, Tensor, int]]
    attention_mask: Optional[Tensor]
    causal_mask: Optional[Any]
    sliding_window: Optional[int] = None
</syntaxhighlight>

=== Backend Selection ===

<syntaxhighlight lang="python">
def select_attention_backend(use_varlen: bool = False) -> str:
    """Return attention backend based on availability / priority order."""
    if HAS_FLASH_ATTENTION:
        if use_varlen:
            return FLASH_VARLEN
        else:
            return FLASH_DENSE
    if HAS_XFORMERS:
        return XFORMERS
    return SDPA
</syntaxhighlight>

== I/O Contract ==

=== Backend Priority ===

{| class="wikitable"
|-
! Priority !! Backend !! Condition !! Best For
|-
| 1 || `flash_varlen` || FlashAttention + packed sequences || Variable-length batches
|-
| 2 || `flash_dense` || FlashAttention available || Dense attention
|-
| 3 || `xformers` || xFormers installed || GPU without FlashAttention
|-
| 4 || `sdpa` || Always (fallback) || CPU or no fused kernels
|}

=== run_attention Signature ===

<syntaxhighlight lang="python">
def run_attention(
    *,
    config: AttentionConfig,
    context: AttentionContext,
    Q: Tensor,  # Query tensor [bsz, n_heads, q_len, head_dim]
    K: Tensor,  # Key tensor [bsz, n_kv_heads, kv_seq_len, head_dim]
    V: Tensor,  # Value tensor [bsz, n_kv_heads, kv_seq_len, head_dim]
) -> Tensor:  # Output [bsz, q_len, n_heads, head_dim]
</syntaxhighlight>

=== AttentionContext Parameters ===

{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| `bsz` || `int` || Batch size
|-
| `q_len` || `int` || Query sequence length
|-
| `kv_seq_len` || `int` || Key/Value sequence length
|-
| `n_heads` || `int` || Number of attention heads
|-
| `head_dim` || `int` || Dimension per head
|-
| `requires_grad` || `bool` || Whether gradients are needed (training vs inference)
|-
| `seq_info` || `Tuple[Tensor, Tensor, int]` || Packed sequence info: (seq_lens, cu_seqlens, max_seqlen)
|-
| `attention_mask` || `Tensor` || Attention mask tensor
|-
| `causal_mask` || `Any` || Causal attention mask
|-
| `sliding_window` || `int` || Sliding window size for local attention
|}

== Usage Examples ==

=== Basic Attention Dispatch ===

<syntaxhighlight lang="python">
from unsloth.utils.attention_dispatch import (
    AttentionConfig, AttentionContext,
    select_attention_backend, run_attention
)
import torch

# Select backend
backend = select_attention_backend(use_varlen=False)

# Configure attention
config = AttentionConfig(
    backend=backend,
    n_kv_heads=8,
    n_groups=4,  # 32 heads / 8 kv_heads = GQA with 4 groups
)

# Create context for this forward pass
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

# Run attention
Q = torch.randn(2, 32, 512, 128, device="cuda")
K = torch.randn(2, 8, 512, 128, device="cuda")
V = torch.randn(2, 8, 512, 128, device="cuda")

output = run_attention(config=config, context=context, Q=Q, K=K, V=V)
# output shape: [2, 512, 32, 128]
</syntaxhighlight>

=== Variable-Length Packed Sequences ===

<syntaxhighlight lang="python">
from unsloth.utils.attention_dispatch import (
    AttentionConfig, AttentionContext,
    select_attention_backend, run_attention
)
import torch

# Select varlen backend for packed sequences
backend = select_attention_backend(use_varlen=True)

# Packed sequence info
seq_lens = torch.tensor([128, 256, 128], device="cuda")  # 3 sequences
cu_seqlens = torch.tensor([0, 128, 384, 512], dtype=torch.int32, device="cuda")
max_seqlen = 256

config = AttentionConfig(
    backend=backend,
    n_kv_heads=8,
    n_groups=4,
)

context = AttentionContext(
    bsz=1,  # Packed as single batch
    q_len=512,  # Total packed length
    kv_seq_len=512,
    n_heads=32,
    head_dim=128,
    requires_grad=True,
    seq_info=(seq_lens, cu_seqlens, max_seqlen),
    attention_mask=None,
    causal_mask=None,
)

output = run_attention(config=config, context=context, Q=Q, K=K, V=V)
</syntaxhighlight>

=== With Sliding Window Attention ===

<syntaxhighlight lang="python">
context = AttentionContext(
    bsz=2,
    q_len=4096,
    kv_seq_len=4096,
    n_heads=32,
    head_dim=128,
    requires_grad=False,  # Inference mode
    seq_info=None,
    attention_mask=None,
    causal_mask=None,
    sliding_window=4096,  # Local attention window
)

output = run_attention(config=config, context=context, Q=Q, K=K, V=V)
</syntaxhighlight>

=== Backend-Specific Kwargs ===

<syntaxhighlight lang="python">
config = AttentionConfig(
    backend="flash_dense",
    n_kv_heads=8,
    n_groups=4,
    flash_dense_kwargs={
        "causal": True,
        "softmax_scale": 1.0 / (128 ** 0.5),
    },
    sdpa_kwargs={
        "dropout_p": 0.1,
        "scale": 1.0 / (128 ** 0.5),
    },
)
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Device_Type|Device Type]] - Hardware detection for backend availability
* [[Unslothai_Unsloth_Model_Registry|Model Registry]] - Model configuration that uses attention dispatch
