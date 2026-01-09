# Implementation: FastGemmaModel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Gemma|https://ai.google.dev/gemma]]
|-
! Domains
| [[domain::Models]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching module for Google Gemma models with fast GEGLU inference and attention.

=== Description ===
This module provides Unsloth-optimized implementations for Gemma models (1.0). It includes fast GEGLU activation inference (using tanh approximation), optimized decoder layer forward passes, and efficient attention with sample packing support.

=== Usage ===
Used automatically when loading Gemma models through `FastLanguageModel.from_pretrained()`. Requires transformers >= 4.38.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/gemma.py unsloth/models/gemma.py]
* '''Lines:''' 1-474

=== Key Functions ===
<syntaxhighlight lang="python">
def fast_geglu_inference(self, X: torch.Tensor) -> torch.Tensor:
    """
    Fast GEGLU inference for Gemma MLP.
    Uses tanh approximation: gelu(gate) * up
    """

def GemmaDecoderLayer_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.Tensor, ...]:
    """Fast forward pass for Gemma decoder layer."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-7b",
    max_seq_length=4096,
    load_in_4bit=True,
)
</syntaxhighlight>

== Gemma Architecture Notes ==

{| class="wikitable"
|-
! Feature !! Description
|-
| GEGLU Activation || Uses GELU-gated linear unit in MLP
|-
| RoPE || Rotary Position Embeddings
|-
| RMSNorm || Root Mean Square normalization
|-
| Minimum Transformers || Requires transformers >= 4.38
|}

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
