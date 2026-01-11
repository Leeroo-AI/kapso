# Implementation: FastCohereModel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Cohere Command R|https://huggingface.co/CohereForAI/c4ai-command-r-plus]]
|-
! Domains
| [[domain::Models]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Optimized patching module for Cohere Command R/R+ models with QK normalization and fast attention.

=== Description ===
This module provides Unsloth-optimized implementations for Cohere models (Command R, Command R+). It includes optimized attention with QK normalization support, fast layernorm inference, and efficient memory management. Cohere models use a unique architecture with QK normalization in the attention layer.

=== Usage ===
Used automatically when loading Cohere models through `FastLanguageModel.from_pretrained()`. The patches are applied to the HuggingFace transformers Cohere implementation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/cohere.py unsloth/models/cohere.py]
* '''Lines:''' 1-526

=== Key Functions ===
<syntaxhighlight lang="python">
def CohereAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    causal_mask: Optional[BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Fast forward pass for Cohere attention with QK normalization.
    """

def fast_layernorm_inference(
    self,
    X: torch.Tensor,
    out_weight: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fast FP32 layernorm for inference.
    Used for QK normalization in Cohere attention.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Usually accessed through FastLanguageModel
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="CohereForAI/c4ai-command-r-plus",
    max_seq_length=4096,
    load_in_4bit=True,
)
# Cohere patches applied automatically
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Attention) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| hidden_states || Tensor || Yes || [batch, seq_len, hidden_dim]
|-
| attention_mask || Tensor || No || Attention mask
|-
| position_ids || Tensor || No || Position indices
|-
| past_key_value || Tuple || No || KV cache for generation
|-
| position_embeddings || Tuple || No || (cos, sin) for RoPE
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| attn_output || Tensor || [batch, seq_len, hidden_dim]
|-
| attn_weights || Tensor/None || Attention weights if output_attentions
|-
| past_key_value || Tuple/None || Updated KV cache if use_cache
|}

== Cohere Architecture Notes ==

Cohere models have unique characteristics handled by this module:

{| class="wikitable"
|-
! Feature !! Description
|-
| QK Normalization || LayerNorm applied to Q and K before attention
|-
| Grouped Query Attention || Multiple query heads per KV head
|-
| Rotary Position Embeddings || RoPE for position encoding
|-
| Logit Softcapping || (Some variants) Soft cap on attention logits
|}

== Usage Examples ==

=== Load and Fine-tune Cohere Model ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="CohereForAI/c4ai-command-r-v01",
    max_seq_length=4096,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Train with SFTTrainer...
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
