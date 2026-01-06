{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Model_Architecture]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Legacy attention mask utilities that convert 2D attention masks to 4D masks with causal masking support and sliding window capabilities, now deprecated in favor of masking_utils.py primitives.

=== Description ===
The modeling_attn_mask_utils module provides the AttentionMaskConverter class and helper functions for transforming attention masks between different dimensionalities. It converts 2D masks (batch_size, sequence_length) to 4D masks (batch_size, 1, query_length, key_value_length) suitable for scaled dot-product attention, with support for causal (unidirectional) masking and sliding window attention. The module includes specialized functions for PyTorch's SDPA (scaled_dot_product_attention) that optimize mask handling by detecting when masks can be omitted in favor of SDPA's is_causal parameter, enabling dispatch to more efficient attention kernels like Flash Attention. All code in this module is marked as deprecated and maintained only for backward compatibility.

=== Usage ===
Use these utilities when working with legacy transformer models that require explicit 4D attention mask construction. The _prepare_4d_causal_attention_mask function creates causal masks for autoregressive models, while _prepare_4d_causal_attention_mask_for_sdpa optimizes masks for PyTorch's SDPA. AttentionMaskConverter.to_4d handles both causal and non-causal mask expansion with optional sliding window support. For new implementations, prefer the more general masking_utils.py primitives instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/modeling_attn_mask_utils.py

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class AttentionMaskConverter:
    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        pass

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        # Convert 2D mask to 4D with causal mask if needed
        pass

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, str] = "cpu",
    ) -> Optional[torch.Tensor]:
        # Create causal 4D mask with optional sliding window
        pass

def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, tuple, list],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    # Prepare 4D causal mask for standard attention
    pass

def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, tuple, list],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    # Prepare optimized mask for SDPA (may return None)
    pass
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_attention_mask,
    _create_4d_causal_attention_mask,
)
</syntaxhighlight>

== I/O Contract ==

=== AttentionMaskConverter.to_4d Inputs ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| attention_mask_2d || torch.Tensor || 2D mask of shape (batch_size, sequence_length) with 1s for attended positions
|-
| query_length || int || Length of the query sequence
|-
| dtype || torch.dtype || Data type for the output mask (e.g., torch.float32)
|-
| key_value_length || int (optional) || Length of key-value sequence (required if causal)
|}

=== AttentionMaskConverter.to_4d Outputs ===
{| class="wikitable"
! Return !! Type !! Description
|-
| mask_4d || torch.Tensor || 4D mask (batch_size, 1, query_length, key_value_length) with large negative values for masked positions
|}

=== _prepare_4d_causal_attention_mask_for_sdpa Inputs ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| attention_mask || torch.Tensor (optional) || 2D or 4D attention mask, or None for no mask
|-
| input_shape || tuple/list || Shape tuple defining (batch_size, query_length)
|-
| inputs_embeds || torch.Tensor || Input embeddings tensor for device/dtype inference
|-
| past_key_values_length || int || Length of cached key-values from previous forward passes
|-
| sliding_window || int (optional) || Window size for sliding window attention
|}

=== _prepare_4d_causal_attention_mask_for_sdpa Outputs ===
{| class="wikitable"
! Return !! Type !! Description
|-
| mask || torch.Tensor or None || 4D mask for SDPA or None if mask can be safely ignored (optimization)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Create causal 4D mask with AttentionMaskConverter
import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

converter = AttentionMaskConverter(is_causal=True)
mask_2d = torch.tensor([[1, 1, 1, 1, 1]])  # All positions attended
mask_4d = converter.to_4d(
    mask_2d,
    query_length=5,
    key_value_length=5,
    dtype=torch.float32
)
print(mask_4d.shape)  # torch.Size([1, 1, 5, 5])

# Example 2: Create causal mask with sliding window
converter_window = AttentionMaskConverter(is_causal=True, sliding_window=3)
mask_4d_window = converter_window.to_causal_4d(
    batch_size=2,
    query_length=8,
    key_value_length=8,
    dtype=torch.float32,
    device="cuda"
)

# Example 3: Prepare mask for model forward pass
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

batch_size, seq_len = 2, 10
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
inputs_embeds = torch.randn(batch_size, seq_len, 768)
attention_mask = torch.ones(batch_size, seq_len)

# Prepare 4D causal mask
causal_mask = _prepare_4d_causal_attention_mask(
    attention_mask=attention_mask,
    input_shape=(batch_size, seq_len),
    inputs_embeds=inputs_embeds,
    past_key_values_length=0
)

# Example 4: Prepare optimized mask for SDPA
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

# This may return None if mask can be safely ignored
sdpa_mask = _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask=attention_mask,
    input_shape=(batch_size, seq_len),
    inputs_embeds=inputs_embeds,
    past_key_values_length=0
)

if sdpa_mask is None:
    print("Mask optimized away - SDPA will use is_causal=True")

# Example 5: Handle left-padding case with unmasking
converter = AttentionMaskConverter(is_causal=True)
# Left-padded sequence: [PAD, PAD, tok1, tok2, tok3]
attention_mask_padded = torch.tensor([[0, 0, 1, 1, 1]])
mask_4d = converter.to_4d(
    attention_mask_padded,
    query_length=5,
    key_value_length=5,
    dtype=torch.float32
)

# Unmask fully masked rows for SDPA memory-efficient attention
min_dtype = torch.finfo(torch.float32).min
unmasked = AttentionMaskConverter._unmask_unattended(mask_4d, min_dtype)

# Example 6: Create pure causal mask without attention_mask
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

pure_causal = _create_4d_causal_attention_mask(
    input_shape=(2, 10),
    dtype=torch.float32,
    device="cpu",
    past_key_values_length=0
)
print(pure_causal.shape)  # torch.Size([2, 1, 10, 10])
</syntaxhighlight>

== Related Pages ==
* (Empty)
