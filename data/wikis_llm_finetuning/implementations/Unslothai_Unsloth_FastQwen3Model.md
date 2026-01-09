# Implementation: FastQwen3Model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Qwen|https://qwenlm.github.io/]]
|-
! Domains
| [[domain::Models]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching module for Qwen 3 models with RMSNorm and optimized attention.

=== Description ===
This module provides Unsloth-optimized implementations for Qwen 3 models. It includes optimized attention forward passes with grouped query attention, dynamic RoPE extension, and efficient inference mode. Requires transformers >= 4.50.3.

=== Usage ===
Used automatically when loading Qwen 3 models through `FastLanguageModel.from_pretrained()`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/qwen3.py unsloth/models/qwen3.py]
* '''Lines:''' 1-457

=== Key Functions ===
<syntaxhighlight lang="python">
def Qwen3Attention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Fast forward pass for Qwen 3 attention."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

== Qwen 3 Architecture Notes ==

{| class="wikitable"
|-
! Feature !! Description
|-
| QK Normalization || RMSNorm on Q and K before attention
|-
| Grouped Query Attention || Multiple Q heads per KV head
|-
| RoPE || Rotary Position Embeddings
|-
| SwiGLU || Swish-gated linear unit in MLP
|-
| Minimum Transformers || Requires transformers >= 4.50.3
|}

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
