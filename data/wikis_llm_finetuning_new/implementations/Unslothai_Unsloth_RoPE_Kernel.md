# Implementation: RoPE Kernel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::GPU_Optimization]], [[domain::Position_Encoding]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

The RoPE (Rotary Position Embedding) kernel provides high-performance Triton implementations for applying rotary position embeddings to query and key tensors in transformer attention mechanisms. RoPE encodes positional information by rotating pairs of elements in the embedding space, enabling the model to learn relative position relationships.

The implementation includes:
* '''Fast RoPE kernel''' for single tensor (Q or K) processing with grouped head optimization
* '''Combined Q/K kernel''' for simultaneous processing of query and key tensors
* '''Slow/inplace fallback''' for compatibility scenarios
* Support for custom rope embedding indices (used in TRL for specific position mappings)

The rotary embedding formula is:
<code>Q_rot = Q * cos + rotate_half(Q) * sin</code>

where <code>rotate_half</code> swaps the first and second halves of the head dimension with negation.

== Code Reference ==

'''File:''' <code>unsloth/kernels/rope_embedding.py</code>

=== Combined Q/K Kernel ===

<syntaxhighlight lang="python">
def _rope_embedding_QK(
    Q,
    Q_batch_stride,
    Q_head_stride,
    Q_seq_stride,
    K,
    K_batch_stride,
    K_head_stride,
    K_seq_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    rope_embedding_indices,
    seqlen,
    head_dim: tl.constexpr,
    n_heads_K: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    HAS_ROPE_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_position = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # Load sin/cos values (optionally using custom rope indices)
    if HAS_ROPE_INDICES:
        rot_position = tl.load(rope_embedding_indices + row_position).to(tl.int32)
    else:
        rot_position = row_position % seqlen

    # Apply rotation to Q
    q0 = tl.load(q_ptr + col_offsets, mask=mask, other=0)
    q1 = tl.load(q_ptr + half_head_dim + col_offsets, mask=mask, other=0)
    tl.store(q_ptr + col_offsets, q0 * cos1 - q1 * sin1, mask=mask)
    tl.store(q_ptr + half_head_dim + col_offsets, q1 * cos1 + q0 * sin1, mask=mask)

    # Apply rotation to K (if head_position < n_heads_K)
    ...
</syntaxhighlight>

=== Single Tensor Kernel with Group Optimization ===

<syntaxhighlight lang="python">
ROPE_GROUP_SIZE: int = 4

def _rope_embedding(
    Q,
    Q_row_stride: tl.constexpr,
    cos,
    cos_row_stride: tl.constexpr,
    sin,
    sin_row_stride: tl.constexpr,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Calculates RoPE Embedding quickly
    RoPE is Q * cos + rotate_half(Q) * sin
    """
    # Process heads in groups of ROPE_GROUP_SIZE for better performance
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    for k in range(head_start, head_end):
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
</syntaxhighlight>

=== PyTorch Autograd Functions ===

<syntaxhighlight lang="python">
class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.reshape(batch * seq_len, n_heads * head_dim)
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
        n_groups = div + (mod != 0)  # ceil division by ROPE_GROUP_SIZE

        _rope_embedding[(n_rows, n_groups)](...)
        return Q.reshape(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        # Backward pass uses negated sin values
        _rope_embedding[...](BACKWARD_PASS=True)
        return (dY, None, None)


class Fast_RoPE_Embedding_QK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, cos, sin, rope_indices):
        # Process Q and K simultaneously
        _rope_embedding_QK[(batch * seq_len, n_heads_Q)](...)
        return (Q_out, K_out)
</syntaxhighlight>

== I/O Contract ==

=== fast_rope_embedding ===

'''Signature:'''
<syntaxhighlight lang="python">
def fast_rope_embedding(
    Q,
    K,
    cos,
    sin,
    rope_embedding_indices=None,
) -> Tuple[torch.Tensor, torch.Tensor]
</syntaxhighlight>

'''Inputs:'''
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| <code>Q</code> || <code>torch.Tensor</code> || Query tensor of shape <code>(batch, n_heads_q, seq_len, head_dim)</code>
|-
| <code>K</code> || <code>torch.Tensor</code> || Key tensor of shape <code>(batch, n_heads_k, seq_len, head_dim)</code>
|-
| <code>cos</code> || <code>torch.Tensor</code> || Cosine cache of shape <code>(max_seq_len, head_dim//2)</code> or squeezable
|-
| <code>sin</code> || <code>torch.Tensor</code> || Sine cache of shape <code>(max_seq_len, head_dim//2)</code> or squeezable
|-
| <code>rope_embedding_indices</code> || <code>torch.Tensor</code> or <code>None</code> || Optional custom position indices (int32)
|}

'''Outputs:'''
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| <code>Q_out</code> || <code>torch.Tensor</code> || Rotated query tensor, same shape as input Q
|-
| <code>K_out</code> || <code>torch.Tensor</code> || Rotated key tensor, same shape as input K
|}

'''Constraints:'''
* Head dimension must be even (split in half for rotation)
* <code>seq_len <= cos.shape[0]</code> (cos/sin must cover the sequence length)
* Supports multi-query attention (n_heads_k can differ from n_heads_q)

=== inplace_rope_embedding (Fallback) ===

'''Signature:'''
<syntaxhighlight lang="python">
def inplace_rope_embedding(Q, K, cos, sin, position_ids) -> Tuple[torch.Tensor, torch.Tensor]
</syntaxhighlight>

Used as a fallback when custom position_ids are needed or for compatibility scenarios.

== Usage Examples ==

=== Basic RoPE Application ===

<syntaxhighlight lang="python">
from unsloth.kernels import fast_rope_embedding

# Prepare inputs
batch, seq_len, n_heads, head_dim = 2, 1024, 32, 128
Q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
K = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

# Create rotary embedding cache
max_seq_len = 2048
cos = torch.randn(max_seq_len, head_dim // 2, device="cuda")
sin = torch.randn(max_seq_len, head_dim // 2, device="cuda")

# Apply rotary embeddings
Q_rot, K_rot = fast_rope_embedding(Q, K, cos, sin)
</syntaxhighlight>

=== With Custom Position Indices (TRL Integration) ===

<syntaxhighlight lang="python">
# For cases where position mapping is non-sequential
rope_indices = torch.arange(seq_len, dtype=torch.int32, device="cuda")
rope_indices = rope_indices.unsqueeze(0).expand(batch, -1)  # (batch, seq_len)

Q_rot, K_rot = fast_rope_embedding(Q, K, cos, sin, rope_embedding_indices=rope_indices)
</syntaxhighlight>

=== Multi-Query Attention Support ===

<syntaxhighlight lang="python">
# GQA/MQA scenario: fewer K heads than Q heads
n_heads_q = 32
n_heads_k = 8  # Grouped Query Attention with 4 groups

Q = torch.randn(batch, n_heads_q, seq_len, head_dim, dtype=torch.float16, device="cuda")
K = torch.randn(batch, n_heads_k, seq_len, head_dim, dtype=torch.float16, device="cuda")

# Kernel handles differing head counts automatically
Q_rot, K_rot = fast_rope_embedding(Q, K, cos, sin)
</syntaxhighlight>

=== Gradient Computation ===

<syntaxhighlight lang="python">
Q = torch.randn(2, 32, 512, 128, dtype=torch.float16, device="cuda", requires_grad=True)
K = torch.randn(2, 32, 512, 128, dtype=torch.float16, device="cuda", requires_grad=True)

Q_rot, K_rot = fast_rope_embedding(Q, K, cos, sin)

# Backward pass uses negated sin for inverse rotation
loss = Q_rot.sum() + K_rot.sum()
loss.backward()  # Gradients flow through custom backward kernel
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Kernel_Utils]] - Utility functions including <code>calculate_settings</code> and <code>torch_gpu_device</code>
* [[Unslothai_Unsloth_RMSNorm_Kernel]] - RMS layer normalization kernel
* [[Unslothai_Unsloth_SwiGLU_Kernel]] - SwiGLU activation function kernel
