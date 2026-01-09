# Implementation: FastFalconH1Model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Models]], [[domain::Hybrid]], [[domain::Mamba]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching module for Falcon H1 hybrid Mamba-Attention models.

=== Description ===
This module provides Unsloth-optimized implementations for Falcon H1 models, which use a hybrid architecture combining Mamba state-space layers with traditional attention layers. The patching includes optimized attention forward passes and support for the FalconHybridMambaAttentionDynamicCache.

=== Usage ===
Used automatically when loading Falcon H1 models through `FastLanguageModel.from_pretrained()`. Requires transformers >= 4.53.0.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/falcon_h1.py unsloth/models/falcon_h1.py]
* '''Lines:''' 1-764

=== Key Functions ===
<syntaxhighlight lang="python">
def FalconH1Attention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Fast forward pass for Falcon H1 attention layers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="tiiuae/Falcon-H1-1B",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

== Falcon H1 Architecture Notes ==

{| class="wikitable"
|-
! Feature !! Description
|-
| Hybrid Architecture || Combines Mamba SSM layers with Attention layers
|-
| Dynamic Cache || Uses FalconHybridMambaAttentionDynamicCache
|-
| RoPE || Rotary Position Embeddings for attention layers
|-
| Minimum Transformers || Requires transformers >= 4.53.0
|}

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
