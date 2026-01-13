# Implementation: Qwen3 MoE Model

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

The Qwen3 MoE model implementation (`unsloth/models/qwen3_moe.py`) provides Unsloth-optimized support for Alibaba's Qwen3 Mixture of Experts models. This implementation extends the base Qwen3 architecture with sparse MoE routing, enabling efficient scaling through selective expert activation.

Key features include:
* '''Sparse MoE Block''': Optimized `Qwen3MoeSparseMoeBlock_fast_forward` with top-k expert selection
* '''Softmax Routing''': Router logits processed through softmax for expert weight computation
* '''Top-K Expert Selection''': Configurable number of experts activated per token
* '''Expert Masking''': One-hot encoding for efficient expert indexing
* '''Shared Attention''': Reuses `Qwen3Attention_fast_forward` from base Qwen3 implementation
* '''Router Logits Output''': Optional output of routing decisions for analysis/loss computation
* '''Fast SwiGLU for Experts''': Each expert MLP uses the optimized `fast_swiglu_inference`

The implementation builds on top of the Qwen3 base model and inherits its attention optimizations.

== Code Reference ==

'''File Path''': `unsloth/models/qwen3_moe.py`

'''Main Classes and Functions''':

* `FastQwen3MoeModel` - Main model class extending `FastQwen3Model`
* `Qwen3MoeSparseMoeBlock_fast_forward` - Optimized sparse MoE routing and computation
* `Qwen3MoeDecoderLayer_fast_forward` - Decoder layer with MoE MLP replacement
* `Qwen3Attention_fast_forward` - Reused from base Qwen3 (via import)

'''Key Dependencies''':

<syntaxhighlight lang="python">
from .llama import *
from .qwen3 import (
    Qwen3Attention_fast_forward,
    FastQwen3Model,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeMLP,
    Qwen3MoeDecoderLayer,
    Qwen3MoeModel,
    Qwen3MoeForCausalLM,
)
</syntaxhighlight>

'''MoE Routing Logic''':

<syntaxhighlight lang="python">
def Qwen3MoeSparseMoeBlock_fast_forward(self, X, temp_gate=None, temp_up=None):
    bsz, seq_len, hd = X.shape
    X = X.view(-1, hd)

    # Compute router logits with optimized linear forward
    router_logits = fast_linear_forward(self.gate_proj, X, out=temp_gate)

    # Softmax routing weights
    routing_weights = torch_nn_functional_softmax(
        router_logits, dim=-1, dtype=torch.float32
    )

    # Select top-k experts
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.top_k, dim=-1
    )
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    # One-hot expert mask for indexing
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.num_experts
    ).permute(2, 1, 0)

    # Loop over experts and compute outputs
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = X[None, top_x].reshape(-1, hd)
        current_X = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_X.index_add_(0, top_x, current_X.to(X.dtype))

    return final_X.reshape(bsz, seq_len, hd), router_logits
</syntaxhighlight>

== I/O Contract ==

'''FastQwen3MoeModel.from_pretrained()'''

{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| model_name || str || "Qwen/Qwen3-7B" || HuggingFace model identifier or path
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

'''Qwen3MoeSparseMoeBlock_fast_forward()'''

{| class="wikitable"
|-
! Input !! Type !! Description
|-
| X || torch.Tensor || Input tensor of shape (batch, seq_len, hidden_dim)
|-
| temp_gate || torch.Tensor || Optional pre-allocated buffer for gate projection
|-
| temp_up || torch.Tensor || Optional pre-allocated buffer for up projection
|}

'''Returns''': Tuple of (output_tensor, router_logits)

'''Qwen3MoeDecoderLayer_fast_forward()'''

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
| output_router_logits || bool || Whether to return routing decisions
|-
| past_key_value || Tuple[torch.Tensor] || Optional cached KV pairs
|-
| position_embeddings || Tuple[torch.Tensor] || Pre-computed cos/sin for RoPE
|}

'''Returns''': Tuple of (hidden_states, [self_attn_weights], [router_logits], [present_key_value])

== Usage Examples ==

'''Basic Model Loading''':

<syntaxhighlight lang="python">
from unsloth import FastQwen3MoeModel

# Load Qwen3 MoE model with 4-bit quantization
model, tokenizer = FastQwen3MoeModel.from_pretrained(
    model_name="Qwen/Qwen2.5-MoE-A14B",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)
</syntaxhighlight>

'''Fine-tuning with LoRA''':

<syntaxhighlight lang="python">
from unsloth import FastQwen3MoeModel

model, tokenizer = FastQwen3MoeModel.from_pretrained(
    model_name="Qwen/Qwen2.5-MoE-A14B",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply LoRA - can target both attention and expert MLPs
model = FastQwen3MoeModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
</syntaxhighlight>

'''Accessing Router Logits''':

<syntaxhighlight lang="python">
# Enable router logits output for analysis or auxiliary loss
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_router_logits=True,
)

# Router logits can be used for:
# - Load balancing loss computation
# - Expert utilization analysis
# - Routing pattern visualization
router_logits = outputs.router_logits
</syntaxhighlight>

'''Pre-patch Application''':

<syntaxhighlight lang="python">
# The pre_patch method is called automatically
FastQwen3MoeModel.pre_patch()

# This patches:
# - Qwen3MoeAttention.forward -> Qwen3Attention_fast_forward
# - Qwen3MoeSparseMoeBlock.forward -> Qwen3MoeSparseMoeBlock_fast_forward
# - Qwen3MoeMLP.forward -> fast_swiglu_inference
# - Qwen3MoeDecoderLayer.forward -> Qwen3MoeDecoderLayer_fast_forward
# - Qwen3MoeModel.forward -> LlamaModel_fast_forward
# - Qwen3MoeForCausalLM.forward -> CausalLM_fast_forward
# - Qwen3MoeRotaryEmbedding -> LlamaRotaryEmbedding
</syntaxhighlight>

'''MoE Architecture Details''':

<syntaxhighlight lang="python">
# Qwen3 MoE uses:
# - num_experts: Total number of expert MLPs
# - top_k: Number of experts activated per token
# - gate_proj: Router projection for computing expert scores
# - experts: List of Qwen3MoeMLP modules

# Each expert MLP uses fast_swiglu_inference for efficiency
# Expert outputs are weighted by normalized routing weights
</syntaxhighlight>

== Related Pages ==

* [[Unslothai_Unsloth_Qwen3_Model]] - Base Qwen3 implementation (attention shared)
* [[Unslothai_Unsloth_Llama_Model]] - Base Llama implementation
* [[Unslothai_Unsloth_Architecture]] - Overview of Unsloth's optimization architecture
* [[Mixture_of_Experts]] - Background on MoE architectures
* [[Sparse_Routing]] - Expert routing mechanisms
