# Implementation: Granite Model

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

The Granite model implementation (`unsloth/models/granite.py`) provides Unsloth-optimized support for IBM's Granite language models. Granite models feature a modified LLaMA-style architecture with distinctive residual multipliers and embedding scaling, designed for enterprise and research applications.

Key features include:
* '''Residual Multiplier''': Implements Granite's unique `residual_multiplier` for scaled residual connections
* '''Embedding Multiplier''': Scales input embeddings via `embedding_multiplier` configuration
* '''Attention Dropout Support''': Configurable attention dropout during training
* '''Custom Softmax Scaling''': Uses model-specific `scaling` parameter for attention softmax
* '''Tied Weight Handling''': Properly manages weight tying between embeddings and LM head
* '''BitsAndBytes Integration''': Full support for 4-bit quantization with proper dtype casting
* '''Position Embeddings Required''': Enforces position embeddings in attention forward pass

The implementation requires transformers version 4.45.0 or higher.

== Code Reference ==

'''File Path''': `unsloth/models/granite.py`

'''Main Classes and Functions''':

* `FastGraniteModel` - Main model class extending `FastLlamaModel`
* `GraniteRotaryEmbedding` - Custom RoPE implementation extending `LlamaRotaryEmbedding`
* `GraniteAttention_fast_forward` - Optimized attention with custom scaling
* `GraniteDecoderLayer_fast_forward` - Decoder layer with residual multiplier support
* `GraniteAttention_fast_forward_inference` - Inference-optimized attention path
* `GraniteModel_fast_forward_inference` - Full model inference with embedding/residual multipliers
* `patched_init` - Decorator to pass config through decoder layers

'''Key Dependencies''':

<syntaxhighlight lang="python">
from .llama import *
from .mistral import *
from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear4bit as Peft_Linear4bit
from transformers.models.granite.modeling_granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteModel,
    GraniteForCausalLM,
)
</syntaxhighlight>

'''Residual Multiplier in Decoder Layer''':

<syntaxhighlight lang="python">
# Granite uses scaled residual connections
residual_multiplier = (
    self.residual_multiplier
    if hasattr(self, "residual_multiplier")
    else self.config.residual_multiplier
)

# Applied after attention
hidden_states = torch.add(residual, hidden_states, alpha=residual_multiplier)

# Applied after MLP
hidden_states = torch.add(residual, hidden_states, alpha=residual_multiplier)
</syntaxhighlight>

== I/O Contract ==

'''FastGraniteModel.from_pretrained()'''

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

'''GraniteAttention_fast_forward()'''

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
| position_embeddings || Tuple[torch.Tensor] || '''Required''' - Pre-computed cos/sin for RoPE
|}

'''Returns''': Tuple of (attn_output, attn_weights, past_key_value)

'''Configuration Parameters''':

{| class="wikitable"
|-
! Config Parameter !! Description
|-
| residual_multiplier || Scaling factor for residual connections (e.g., 0.0625)
|-
| embedding_multiplier || Scales embeddings after token lookup
|-
| attention_dropout || Dropout probability during training
|-
| scaling || Softmax scaling factor for attention (replaces 1/sqrt(d))
|}

== Usage Examples ==

'''Basic Model Loading''':

<syntaxhighlight lang="python">
from unsloth import FastGraniteModel

# Load IBM Granite model with 4-bit quantization
model, tokenizer = FastGraniteModel.from_pretrained(
    model_name="ibm-granite/granite-3b-code-instruct",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)
</syntaxhighlight>

'''Fine-tuning with LoRA''':

<syntaxhighlight lang="python">
from unsloth import FastGraniteModel

model, tokenizer = FastGraniteModel.from_pretrained(
    model_name="ibm-granite/granite-8b-code-instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply LoRA for efficient fine-tuning
model = FastGraniteModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
</syntaxhighlight>

'''Post-patch Operations''':

<syntaxhighlight lang="python">
# FastGraniteModel.post_patch handles:
# 1. Embedding matrix reconstruction for torch.compile compatibility
# 2. LM head weight management
# 3. Tied weight handling (embed_tokens <-> lm_head)
# 4. BitsAndBytes dtype correction
# 5. RoPE cache dtype alignment

model, tokenizer = FastGraniteModel.post_patch(model, tokenizer)
</syntaxhighlight>

'''Pre-patch Application''':

<syntaxhighlight lang="python">
# The pre_patch method is called automatically
FastGraniteModel.pre_patch()

# This patches:
# - GraniteAttention.forward -> GraniteAttention_fast_forward
# - GraniteSdpaAttention.forward -> GraniteAttention_fast_forward
# - GraniteFlashAttention2.forward -> GraniteAttention_fast_forward
# - GraniteDecoderLayer.forward -> GraniteDecoderLayer_fast_forward
# - GraniteModel.forward -> LlamaModel_fast_forward
# - GraniteForCausalLM.forward -> CausalLM_fast_forward
# - GraniteForCausalLM.__init__ -> patched_init (for config propagation)
# - GraniteRotaryEmbedding replacement
</syntaxhighlight>

'''Inference with Embedding Scaling''':

<syntaxhighlight lang="python">
# During inference, the model applies embedding_multiplier automatically:
# hidden_states = self.model.embed_tokens(input_ids)
# hidden_states *= self.model.embedding_multiplier

inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
code = tokenizer.decode(outputs[0], skip_special_tokens=True)
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Llama_Model]] - Base Llama implementation that Granite extends
* [[Unslothai_Unsloth_Mistral_Model]] - Related Mistral implementation (imported)
* [[Unslothai_Unsloth_Architecture]] - Overview of Unsloth's optimization architecture
* [[Unslothai_Unsloth_Attention_Dispatch]] - Attention backend selection system
* [[IBM_Granite_Models]] - Background on IBM Granite architecture
