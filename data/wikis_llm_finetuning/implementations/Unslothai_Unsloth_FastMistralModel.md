# Implementation: FastMistralModel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|Mistral|https://arxiv.org/abs/2310.06825]]
|-
! Domains
| [[domain::Models]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching module for Mistral models with sliding window attention and dynamic RoPE extension.

=== Description ===
This module provides Unsloth-optimized implementations for Mistral models (7B, 8x7B MoE variants). It includes optimized attention with sliding window support, dynamic RoPE embedding extension, and packed sequence boundary masking.

=== Usage ===
Used automatically when loading Mistral models through `FastLanguageModel.from_pretrained()`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/mistral.py unsloth/models/mistral.py]
* '''Lines:''' 1-469

=== Key Functions ===
<syntaxhighlight lang="python">
def MistralAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Fast forward pass for Mistral attention.
    Supports sliding window attention and dynamic RoPE extension.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-v0.3",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

== Mistral Architecture Notes ==

{| class="wikitable"
|-
! Feature !! Description
|-
| Sliding Window Attention || Default 4096 token window
|-
| Grouped Query Attention || Multiple Q heads per KV head
|-
| RoPE || Rotary Position Embeddings with dynamic extension
|-
| SwiGLU || Swish-gated linear unit in MLP
|}

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
