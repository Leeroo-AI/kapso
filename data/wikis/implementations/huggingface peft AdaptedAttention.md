{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|LLaMA-Adapter|https://arxiv.org/abs/2303.16199]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Prompt_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Attention layer wrapper that injects learnable adaption prompts into key-value attention, implementing LLaMA-Adapter style prefix tuning.

=== Description ===

AdaptedAttention wraps LLaMA/Mistral attention modules and injects trainable prompt tokens. It stores adaption_prompt (learnable embeddings) and adaption_gate (zero-initialized scalar). During forward, it computes adapter keys/values from the prompt, calculates attention scores between queries and adapter keys, applies the gate, and adds the result to original attention output. AdaptedAttentionGPT provides equivalent functionality for GPT-2 architecture.

=== Usage ===

Use AdaptedAttention when applying LLaMA-Adapter style fine-tuning. The module wraps existing attention layers and is created automatically by AdaptionPromptModel. Supports LLaMA, Mistral (with grouped query attention), and GPT-2 models.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adaption_prompt/layer.py src/peft/tuners/adaption_prompt/layer.py]
* '''Lines:''' 1-237

=== Signature ===
<syntaxhighlight lang="python">
class _BaseAdaptedAttention(nn.Module):
    """Base class for adapted attention modules."""
    def __init__(
        self,
        model_type: str,
        adapter_len: int,
        model,
        target_dtype=torch.float32,
    ):
        """
        Args:
            model_type: Transformer model type (llama, mistral, gpt2)
            adapter_len: Number of adapter tokens
            model: Original attention module to wrap
        """

class AdaptedAttention(_BaseAdaptedAttention):
    """Wraps LLaMA/Mistral attention with adaption prompts."""
    def forward(self, **kwargs):
        """Forward with injected adapter tokens."""

class AdaptedAttentionGPT(_BaseAdaptedAttention):
    """Wraps GPT-2 attention with adaption prompts."""
    def forward(self, hidden_states, ...):
        """Forward with injected adapter tokens."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.adaption_prompt.layer import AdaptedAttention, AdaptedAttentionGPT
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_type || str || Yes || "llama", "mistral", or "gpt2"
|-
| adapter_len || int || Yes || Number of adapter tokens
|-
| model || nn.Module || Yes || Original attention to wrap
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Attention output + adapter contribution
|}

== Usage Examples ==

=== LLaMA-Adapter Forward Pass ===
<syntaxhighlight lang="python">
# AdaptedAttention is created automatically by AdaptionPromptModel
# The forward computation:

# 1. Get original attention output
output = self.model(**kwargs)

# 2. Project adapter prompt to keys/values
key = self.model.k_proj(self.adaption_prompt)
value = self.model.v_proj(self.adaption_prompt)

# 3. Recompute query states
query_states = compute_query_states(self.model, **kwargs)

# 4. Compute adapter attention
scores = query_states @ adapter_k.T / sqrt(head_dim)
scores = self.adaption_gate * softmax(scores)
adapter_output = scores @ adapter_v

# 5. Add to original output
output = output + adapter_output
</syntaxhighlight>

=== Zero-Init Gating ===
<syntaxhighlight lang="python">
# Gate is initialized to 0, allowing gradual contribution
self.adaption_gate = nn.Parameter(torch.zeros(1))

# During training, gate learns to scale adapter influence
# This provides stable training initialization
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
