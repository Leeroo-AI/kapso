# Implementation: Falcon H1 Model

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

The Falcon H1 model implementation (`unsloth/models/falcon_h1.py`) provides Unsloth-optimized support for TII's Falcon H1 hybrid architecture. Falcon H1 is a unique hybrid model that combines Mamba (State Space Model) blocks with Transformer attention layers, enabling efficient long-context processing while maintaining strong performance.

Key features include:
* '''Hybrid Mamba + Transformer Architecture''': Combines SSM (State Space Model) blocks with attention for efficient sequence processing
* '''Key Multiplier Support''': Implements Falcon H1's key state scaling mechanism (`key_multiplier`)
* '''Dual Output Multipliers''': Supports separate `attn_out_multiplier` and `ssm_out_multiplier` for combining attention and Mamba outputs
* '''Embedding Multiplier''': Scales input embeddings via `embedding_multiplier` configuration
* '''MLP Multipliers''': Configurable `gate_multiplier` and `down_multiplier` for feed-forward layers
* '''Specialized KV Cache''': Uses `FalconHybridMambaAttentionDynamicCache` for hybrid caching
* '''Optimized Inference Path''': Custom inference forward pass handling both Mamba and attention states

The implementation requires transformers version 4.53.0 or higher.

== Code Reference ==

'''File Path''': `unsloth/models/falcon_h1.py`

'''Main Classes and Functions''':

* `FastFalconH1Model` - Main model class extending `FastLlamaModel`
* `FalconH1Attention_fast_forward` - Optimized attention with key multiplier
* `FalconH1DecoderLayer_fast_forward` - Hybrid decoder combining Mamba and attention
* `FalconH1Attention_fast_forward_inference` - Inference-optimized attention with paged KV cache
* `_FalconH1_fast_forward_inference` - Factory function for customizable inference
* `_fast_prepare_inputs_for_generation` - Specialized input preparation for hybrid generation

'''Key Dependencies''':

<syntaxhighlight lang="python">
from .llama import *
from transformers.models.falcon_h1.modeling_falcon_h1 import (
    FalconH1Attention,
    FalconH1DecoderLayer,
    FalconH1Model,
    FalconH1ForCausalLM,
    FalconHybridMambaAttentionDynamicCache,
)
</syntaxhighlight>

'''Hybrid Architecture in Decoder Layer''':

<syntaxhighlight lang="python">
# The decoder layer combines both Mamba and Attention outputs
mamba_hidden_states = self.mamba(
    hidden_states=hidden_states,
    cache_params=past_key_value,
    cache_position=cache_position,
    attention_mask=mamba_attention_mask,
)
mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

attention_hidden_states, self_attn_weights, present_key_value = self.self_attn(...)
attention_hidden_states = attention_hidden_states * self.attn_out_multiplier

# Combine both pathways
hidden_states = mamba_hidden_states + attention_hidden_states
</syntaxhighlight>

== I/O Contract ==

'''FastFalconH1Model.from_pretrained()'''

{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| model_name || str || "Qwen/FalconH1-7B" || HuggingFace model identifier or path
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
|-
| trust_remote_code || bool || False || Allow custom code execution
|}

'''FalconH1DecoderLayer_fast_forward()'''

{| class="wikitable"
|-
! Input !! Type !! Description
|-
| hidden_states || torch.Tensor || Input tensor of shape (batch, seq_len, hidden_dim)
|-
| attention_mask || torch.Tensor || Attention mask for transformer blocks
|-
| mamba_attention_mask || torch.Tensor || Separate mask for Mamba blocks
|-
| position_ids || torch.LongTensor || Position indices for RoPE
|-
| cache_position || torch.LongTensor || Cache position for stateful generation
|-
| past_key_value || Tuple || Combined Mamba and attention cache
|-
| position_embeddings || Tuple[torch.Tensor] || Pre-computed cos/sin for RoPE
|}

'''Returns''': Tuple of (hidden_states, [self_attn_weights], [present_key_value])

'''Configuration Multipliers''':

{| class="wikitable"
|-
! Config Parameter !! Description
|-
| embedding_multiplier || Scales input embeddings after token lookup
|-
| key_multiplier || Scales key states in attention (K = K * key_multiplier)
|-
| attn_out_multiplier || Scales attention output before combining
|-
| ssm_out_multiplier || Scales Mamba SSM output before combining
|-
| mlp_multipliers || Tuple of (gate_multiplier, down_multiplier) for MLP
|}

== Usage Examples ==

'''Basic Model Loading''':

<syntaxhighlight lang="python">
from unsloth import FastFalconH1Model

# Load Falcon H1 hybrid model with 4-bit quantization
model, tokenizer = FastFalconH1Model.from_pretrained(
    model_name="tiiuae/Falcon-H1-7B",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)
</syntaxhighlight>

'''Fine-tuning with LoRA''':

<syntaxhighlight lang="python">
from unsloth import FastFalconH1Model

model, tokenizer = FastFalconH1Model.from_pretrained(
    model_name="tiiuae/Falcon-H1-7B",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply LoRA - targets both attention and Mamba components
model = FastFalconH1Model.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
</syntaxhighlight>

'''Inference with Hybrid Cache''':

<syntaxhighlight lang="python">
# The model handles hybrid caching automatically during generation
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# Generation uses FalconHybridMambaAttentionDynamicCache internally
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    use_cache=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
</syntaxhighlight>

'''Pre-patch Application''':

<syntaxhighlight lang="python">
# The pre_patch method patches both attention and integrates with Mamba
FastFalconH1Model.pre_patch()

# This patches:
# - FalconH1Attention.forward -> FalconH1Attention_fast_forward
# - FalconH1DecoderLayer.forward -> FalconH1DecoderLayer_fast_forward
# - FalconH1Model.forward -> LlamaModel_fast_forward
# - FalconH1ForCausalLM.forward -> CausalLM_fast_forward (with hybrid inference)
# - FalconH1RotaryEmbedding -> LlamaRotaryEmbedding
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Llama_Model]] - Base Llama implementation
* [[Unslothai_Unsloth_Architecture]] - Overview of Unsloth's optimization architecture
* [[Unslothai_Unsloth_Attention_Dispatch]] - Attention backend selection system
* [[Mamba_State_Space_Models]] - Background on SSM architectures
* [[Hybrid_Transformer_Architectures]] - Overview of hybrid model designs
