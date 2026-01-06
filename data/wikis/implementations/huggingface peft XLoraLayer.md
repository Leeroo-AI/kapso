{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|X-LoRA|https://arxiv.org/abs/2402.07148]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Mixture_of_Experts]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

X-LoRA layer wrapper that dynamically routes and scales multiple LoRA adapters using learned per-token scalings for mixture-of-experts style adaptation.

=== Description ===

XLoraLayer wraps existing LoRA layers to enable dynamic mixing of multiple LoRA adapters. A classifier network generates per-token, per-layer scaling factors that weight each adapter's contribution. The layer supports top-k selection to activate only the most relevant adapters per token, and optional softmax normalization over selected adapters. This enables a single model to leverage multiple specialized LoRA experts dynamically.

=== Usage ===

Use X-LoRA when you have multiple task-specific LoRA adapters and want to combine them dynamically based on input. X-LoRA is particularly effective for handling diverse inputs that may benefit from different combinations of expertise. The classifier learns to route tokens to appropriate experts during training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/xlora/layer.py src/peft/tuners/xlora/layer.py]
* '''Lines:''' 1-238

=== Signature ===
<syntaxhighlight lang="python">
class XLoraLayer:
    """
    X-LoRA layer wrapper for dynamic adapter mixing.

    Args:
        model: The XLoraModel parent
        target: The LoraLayer being wrapped
        target_forward: Original forward method
        layer_number: Layer index for scaling lookup
        config: XLoraConfig with routing parameters
    """

    @staticmethod
    def apply_scalings_to_x(
        x: torch.Tensor,
        scalings_layer: torch.Tensor,
        adapter: int,
    ) -> torch.Tensor:
        """Apply per-token scalings for an adapter."""

    def get_maybe_topk_scalings(self, scalings) -> torch.Tensor:
        """Get scalings with optional top-k and softmax."""

class XLoraLinearLayer(XLoraLayer):
    """X-LoRA for Linear layers."""
    def forward(
        self,
        x: Tensor,
        *args,
        scalings: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward with dynamic LoRA scaling."""

class XLoraEmbeddingLayer(XLoraLayer):
    """X-LoRA for Embedding layers."""

class XLoraConv2dLayer(XLoraLayer):
    """X-LoRA for Conv2d layers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.xlora import XLoraLayer, XLoraConfig, XLoraModel
from peft import XLoraConfig, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || nn.Module || Yes || Parent XLoraModel
|-
| target || LoraLayer || Yes || The LoRA layer to wrap
|-
| layer_number || int || Yes || Layer index for scaling lookup
|-
| config || XLoraConfig || Yes || Configuration with top_k, softmax settings
|-
| scalings || torch.Tensor || No || Per-token scalings [batch, seq, layers, adapters]
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || torch.Tensor || Dynamically weighted LoRA adaptation
|}

== Usage Examples ==

=== Basic X-LoRA Setup ===
<syntaxhighlight lang="python">
from peft import XLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# First, load a model with multiple LoRA adapters
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Configure X-LoRA to mix multiple adapters
config = XLoraConfig(
    peft_type="XLORA",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    adapters=["adapter1", "adapter2", "adapter3"],
    hidden_size=4096,
    xlora_depth=2,             # Classifier depth
    global_scaling_weight=1.0,
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== X-LoRA with Top-K Selection ===
<syntaxhighlight lang="python">
from peft import XLoraConfig, get_peft_model

# Only activate top-k adapters per token
config = XLoraConfig(
    adapters=["math", "code", "writing", "reasoning"],
    target_modules=["q_proj", "v_proj"],
    top_k_lora=2,              # Only use top 2 adapters per token
    enable_softmax_topk=True,  # Softmax over selected adapters
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== X-LoRA Inference ===
<syntaxhighlight lang="python">
# During inference, scalings are computed by the classifier
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
)
# The model automatically routes tokens through appropriate adapters
</syntaxhighlight>

=== Training X-LoRA Classifier ===
<syntaxhighlight lang="python">
# During training, the classifier learns optimal routing
# The LoRA weights can be frozen while training classifier
model.set_xlora_trainability(trainable=True)

# Or train everything together
for param in model.parameters():
    param.requires_grad = True

outputs = model(**batch)
loss = outputs.loss
loss.backward()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
