# Implementation: Cohere Model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Model_Architecture]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

The Cohere model implementation (`unsloth/models/cohere.py`) provides Unsloth-optimized support for Cohere's language models. This implementation extends the base Llama architecture with Cohere-specific features including QK normalization (Query-Key LayerNorm), which is a distinctive characteristic of Cohere models that helps stabilize attention computations.

Key features include:
* '''QK LayerNorm Support''': Implements query and key normalization in the attention mechanism for improved training stability
* '''Optimized Attention Forward Pass''': Custom `CohereAttention_fast_forward` with support for both training and inference paths
* '''Paged Attention for Inference''': Efficient KV cache management with incremental allocation (256-token increments)
* '''Flash Attention Integration''': Supports Flash Attention 2 and SDPA backends through the attention dispatch system
* '''RoPE Extensions''': Dynamic rotary position embedding extension for longer sequences

The implementation requires transformers version 4.42.3 or higher.

== Code Reference ==

'''File Path''': `unsloth/models/cohere.py`

'''Main Classes and Functions''':

* `FastCohereModel` - Main model class extending `FastLlamaModel`
* `CohereAttention_fast_forward` - Optimized attention forward pass with QK normalization
* `CohereDecoderLayer_fast_forward` - Optimized decoder layer with separate training/inference paths
* `CohereAttention_fast_forward_inference` - Specialized inference path with paged attention
* `CohereModel_fast_forward_inference` - Full model inference with KV cache support
* `fast_layernorm_inference` - Optimized layer normalization for inference

'''Key Dependencies''':

<syntaxhighlight lang="python">
from .llama import *
from transformers.models.cohere.modeling_cohere import (
    CohereAttention,
    CohereDecoderLayer,
    CohereModel,
    CohereForCausalLM,
    CohereRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
</syntaxhighlight>

== I/O Contract ==

'''FastCohereModel.from_pretrained()'''

{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| model_name || str || Required || HuggingFace model identifier or path
|-
| max_seq_length || int || 4096 || Maximum sequence length for the model
|-
| dtype || torch.dtype || None || Data type (auto-detected if None)
|-
| load_in_4bit || bool || True || Enable 4-bit quantization via bitsandbytes
|-
| token || str || None || HuggingFace authentication token
|-
| device_map || str || "sequential" || Device placement strategy
|-
| rope_scaling || dict || None || RoPE scaling configuration
|-
| fix_tokenizer || bool || True || Apply tokenizer fixes
|}

'''CohereAttention_fast_forward()'''

{| class="wikitable"
|-
! Input !! Type !! Description
|-
| hidden_states || torch.Tensor || Input tensor of shape (batch, seq_len, hidden_dim)
|-
| attention_mask || torch.Tensor || Optional attention mask
|-
| position_ids || torch.LongTensor || Position indices for RoPE
|-
| past_key_value || Tuple[torch.Tensor] || Optional cached KV pairs
|-
| position_embeddings || Tuple[torch.Tensor] || Pre-computed cos/sin for RoPE
|}

'''Returns''': Tuple of (attn_output, attn_weights, past_key_value)

== Usage Examples ==

'''Basic Model Loading''':

<syntaxhighlight lang="python">
from unsloth import FastCohereModel

# Load Cohere model with 4-bit quantization
model, tokenizer = FastCohereModel.from_pretrained(
    model_name="CohereForAI/c4ai-command-r-v01",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)
</syntaxhighlight>

'''Adding LoRA Adapters''':

<syntaxhighlight lang="python">
from unsloth import FastCohereModel

model, tokenizer = FastCohereModel.from_pretrained(
    model_name="CohereForAI/c4ai-command-r-v01",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply LoRA for efficient fine-tuning
model = FastCohereModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
</syntaxhighlight>

'''Pre-patch Application''':

<syntaxhighlight lang="python">
# The pre_patch method is called automatically but can be invoked manually
FastCohereModel.pre_patch()

# This patches:
# - CohereAttention.forward -> CohereAttention_fast_forward
# - CohereSdpaAttention.forward -> CohereAttention_fast_forward
# - CohereFlashAttention2.forward -> CohereAttention_fast_forward
# - CohereDecoderLayer.forward -> CohereDecoderLayer_fast_forward
# - CohereModel.forward -> LlamaModel_fast_forward
# - CohereForCausalLM.forward -> CausalLM_fast_forward
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Llama_Model]] - Base Llama implementation that Cohere extends
* [[Unslothai_Unsloth_Architecture]] - Overview of Unsloth's optimization architecture
* [[Unslothai_Unsloth_Attention_Dispatch]] - Attention backend selection system
* [[Unslothai_Unsloth_LoRA_Integration]] - LoRA adapter support
