# Implementation: Flex_Attention

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Kernels]], [[domain::GPU_Optimization]], [[domain::Attention]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==
Provides flex attention with logit softcapping specifically optimized for Gemma2 models, supporting both PyTorch compile-based and FlexAttention API implementations.

=== Description ===
The Flex Attention module implements attention mechanisms with logit softcapping, a technique used in Gemma2 models to stabilize attention scores. The implementation provides two pathways:

# '''Compiled Slow Attention''': When FlexAttention is not available (PyTorch < 2.5), uses torch.compile with full graph optimization to perform attention with grouped query attention (GQA) support and logit softcapping via tanh.

# '''FlexAttention API''': When available (PyTorch >= 2.5), leverages PyTorch's native flex_attention API with custom score modification functions for softcapping.

The logit softcapping formula applies: <code>A = t * tanh(A / t)</code> where t is the softcapping temperature from the model config. This prevents attention logits from growing too large during training.

Key optimizations include:
* Grouped query attention expansion for multi-head attention with fewer KV heads
* Query pre-attention scalar from config for proper scaling
* Causal masking with optional sliding window support
* torch.compile integration with epilogue fusion and max autotune

=== Usage ===
This kernel is automatically invoked during attention computation in Gemma2 models when:
* Training or inference with Gemma2-style models that use logit softcapping
* The model config contains <code>attn_logit_softcapping</code> and <code>query_pre_attn_scalar</code> parameters
* Flex attention masks need to be created for causal or sliding window attention

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' unsloth/kernels/flex_attention.py
* '''Lines:''' 1-187

=== Signature ===
<syntaxhighlight lang="python">
def slow_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
    """
    Performs attention with logit softcapping for Gemma2 models.

    Args:
        Q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        K: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        V: Value tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        causal_mask: Causal attention mask
        self: Model self reference containing config
        bsz: Batch size
        q_len: Query sequence length

    Returns:
        Attention output tensor of shape (batch, seq_len, hidden_size)
    """

def create_flex_attention_causal_mask(max_seq_length=8192):
    """
    Creates a causal block mask for flex attention.

    Args:
        max_seq_length: Maximum sequence length (default: 8192)

    Returns:
        Block mask for causal attention
    """

def create_flex_attention_sliding_window_mask(max_seq_length=8192, sliding_window=4096):
    """
    Creates a sliding window causal mask for flex attention.

    Args:
        max_seq_length: Maximum sequence length (default: 8192)
        sliding_window: Sliding window size (default: 4096)

    Returns:
        Block mask for sliding window causal attention
    """

def slow_inference_attention_softcapping(Q, K, V, causal_mask, self, bsz, q_len):
    """
    Non-compiled version of attention with softcapping for inference.
    Optimized for single-token generation with in-place operations.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.kernels.flex_attention import (
    slow_attention_softcapping,
    create_flex_attention_causal_mask,
    create_flex_attention_sliding_window_mask,
    slow_inference_attention_softcapping,
    HAS_FLEX_ATTENTION,
)
</syntaxhighlight>

== I/O Contract ==

=== slow_attention_softcapping ===

{| class="wikitable"
|+ Input Parameters
|-
! Name !! Type !! Shape !! Description
|-
| Q || torch.Tensor || (batch, n_heads, seq_len, head_dim) || Query tensor
|-
| K || torch.Tensor || (batch, n_kv_heads, seq_len, head_dim) || Key tensor
|-
| V || torch.Tensor || (batch, n_kv_heads, seq_len, head_dim) || Value tensor
|-
| causal_mask || torch.Tensor || (seq_len, seq_len) || Causal attention mask with -inf for masked positions
|-
| self || object || - || Model reference with config containing num_attention_heads, num_key_value_heads, query_pre_attn_scalar, attn_logit_softcapping
|-
| bsz || int || - || Batch size
|-
| q_len || int || - || Query sequence length
|}

{| class="wikitable"
|+ Output
|-
! Name !! Type !! Shape !! Description
|-
| output || torch.Tensor || (batch, seq_len, n_heads * head_dim) || Attention output
|}

=== create_flex_attention_causal_mask ===

{| class="wikitable"
|+ Input Parameters
|-
! Name !! Type !! Default !! Description
|-
| max_seq_length || int || 8192 || Maximum sequence length for the mask
|}

{| class="wikitable"
|+ Output
|-
! Name !! Type !! Description
|-
| block_mask || BlockMask || Compiled block mask for flex attention
|}

== Usage Examples ==

=== Basic Attention with Softcapping ===
<syntaxhighlight lang="python">
import torch
from unsloth.kernels.flex_attention import slow_attention_softcapping

# Assuming model with Gemma2 config
class MockConfig:
    num_attention_heads = 16
    num_key_value_heads = 4
    query_pre_attn_scalar = 256
    attn_logit_softcapping = 50.0

class MockSelf:
    config = MockConfig()
    head_dim = 64
    num_key_value_groups = 4

# Create tensors
bsz, seq_len, n_heads, head_dim = 2, 128, 16, 64
n_kv_heads = 4

Q = torch.randn(bsz, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
K = torch.randn(bsz, n_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
V = torch.randn(bsz, n_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)

# Create causal mask
causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float("-inf"), device="cuda"),
    diagonal=1
)

model_self = MockSelf()

# Compute attention
output = slow_attention_softcapping(Q, K, V, causal_mask, model_self, bsz, seq_len)
print(f"Output shape: {output.shape}")  # (2, 128, 1024)
</syntaxhighlight>

=== Creating Flex Attention Masks ===
<syntaxhighlight lang="python">
from unsloth.kernels.flex_attention import (
    create_flex_attention_causal_mask,
    create_flex_attention_sliding_window_mask,
    HAS_FLEX_ATTENTION,
)

if HAS_FLEX_ATTENTION:
    # Standard causal mask for sequences up to 8192 tokens
    causal_mask = create_flex_attention_causal_mask(max_seq_length=8192)

    # Sliding window mask for local attention
    sliding_mask = create_flex_attention_sliding_window_mask(
        max_seq_length=8192,
        sliding_window=4096
    )
else:
    print("FlexAttention requires PyTorch >= 2.5")
</syntaxhighlight>

=== Inference-Optimized Attention ===
<syntaxhighlight lang="python">
from unsloth.kernels.flex_attention import slow_inference_attention_softcapping

# For inference, use the non-compiled version with in-place optimizations
output = slow_inference_attention_softcapping(
    Q, K, V, causal_mask, model_self, bsz, seq_len
)
</syntaxhighlight>

== Implementation Details ==

=== Logit Softcapping ===
The softcapping mechanism prevents attention logits from becoming too large:

<syntaxhighlight lang="python">
# Standard attention: A = softmax(Q @ K.T / sqrt(d))
# With softcapping: A = softmax(t * tanh((Q @ K.T) / t))

s = self.config.query_pre_attn_scalar  # e.g., 256
t = self.config.attn_logit_softcapping  # e.g., 50.0

Q = Q * (s ** -0.5)  # Scale queries
A = torch.matmul(Q, K.transpose(2, 3))
A = t * torch.tanh(A / t)  # Softcapping
A += causal_mask  # Apply mask
A = softmax(A)
</syntaxhighlight>

=== Grouped Query Attention Expansion ===
When n_kv_heads < n_heads, the K and V tensors are expanded:

<syntaxhighlight lang="python">
n_groups = n_heads // n_kv_heads
K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, seq_len, head_dim)
K = K.reshape(bsz, n_heads, seq_len, head_dim)
</syntaxhighlight>

=== torch.compile Options ===
The module uses specific compile options for optimal performance:

<syntaxhighlight lang="python">
torch_compile_options = {
    "epilogue_fusion": True,      # Fuse epilogue operations
    "max_autotune": True,         # Enable autotuning
    "shape_padding": True,        # Pad shapes for efficiency
    "triton.cudagraphs": False,   # Disable CUDA graphs
}
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Triton_Optimization]]
* [[related::Implementation:Unslothai_Unsloth_RoPE_Kernel]]
* [[related::Implementation:Unslothai_Unsloth_LayerNorm_Kernel]]

== See Also ==
* [https://github.com/google/gemma_pytorch Gemma PyTorch Implementation]
* [https://pytorch.org/docs/stable/generated/torch.nn.attention.flex_attention.html PyTorch FlexAttention Documentation]
