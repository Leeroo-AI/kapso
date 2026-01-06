{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|RandLoRA|https://arxiv.org/abs/2502.00987]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Random_Projection]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Model class that creates RandLoRA adapters with shared random projection matrices across all target layers for parameter-efficient fine-tuning.

=== Description ===

RandLoraModel extends BaseTuner to implement Random projection LoRA. The model initializes shared randlora_A and randlora_B matrices once during _pre_injection_hook, which are then used across all adapted layers. The matrices can be dense (Kaiming initialized) or sparse (ternary projections). The model finds the largest layer dimensions to size shared matrices appropriately, ensuring they can be sliced for layers of different sizes.

=== Usage ===

Use RandLoraModel for parameter-efficient adaptation where you want to minimize trainable parameters. The shared random matrices mean only small lambda and gamma vectors are trained per layer. Use projection_prng_key for reproducible initialization and save_projection to control whether random matrices are saved with checkpoints.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/randlora/model.py src/peft/tuners/randlora/model.py]
* '''Lines:''' 1-357

=== Signature ===
<syntaxhighlight lang="python">
def _kaiming_init(tensor_or_shape, generator: torch.Generator) -> torch.Tensor:
    """Kaiming initialization with PRNG generator."""

class RandLoraModel(BaseTuner):
    """
    Creates RandLoRA model from pretrained transformer.

    Args:
        model: The model to be adapted
        config: RandLoraConfig with r, alpha, projection settings
        adapter_name: Name for the adapter (default: "default")

    Attributes:
        prefix: "randlora_"
        randlora_A: Shared BufferDict of A matrices
        randlora_B: Shared BufferDict of B matrices
    """
    prefix = "randlora_"
    tuner_layer_cls = RandLoraLayer

    def _find_dim(self, config) -> tuple[int, int]:
        """Find largest dimensions across target layers."""

    def _init_randlora_A_randlora_B(self, config, adapter_name):
        """Initialize dense random projection matrices."""

    def _init_randlora_A_randlora_B_sparse(self, config, adapter_name, sparsity):
        """Initialize sparse ternary projection matrices."""

    def _pre_injection_hook(self, model, config, adapter_name):
        """Initialize shared matrices before layer injection."""

    def _create_new_module(randlora_config, randlora_A, randlora_B, adapter_name, target, **kwargs):
        """Create RandLoRA module with shared bases."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import RandLoraModel, RandLoraConfig, get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || The pretrained model to adapt
|-
| config || RandLoraConfig || Yes || Configuration with r, alpha, projection_prng_key
|-
| adapter_name || str || No || Name for adapter (default: "default")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward() || ModelOutput || Model output with RandLoRA adaptation
|-
| randlora_A || BufferDict || Shared frozen A matrices
|-
| randlora_B || BufferDict || Shared frozen B matrices
|}

== Usage Examples ==

=== Creating RandLoRA Model ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

config = RandLoraConfig(
    r=32,
    randlora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj"],
    projection_prng_key=42,
)

model = get_peft_model(base_model, config)

# Shared matrices are in model.randlora_A and model.randlora_B
# Per-layer parameters are in each layer's lambda and gamma
</syntaxhighlight>

=== RandLoRA with Reproducible Projections ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model

# Same key = same random matrices
config = RandLoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    projection_prng_key=12345,  # Deterministic
    save_projection=False,      # Regenerate on load
)

model = get_peft_model(model, config)

# When loading, use same key to recreate matrices
</syntaxhighlight>

=== Sparse RandLoRA for Efficiency ===
<syntaxhighlight lang="python">
from peft import RandLoraConfig, get_peft_model

# Sparse ternary projections {-1, 0, 1}
config = RandLoraConfig(
    r=64,
    target_modules=["q_proj", "v_proj"],
    sparse=True,              # Sparsity = 3
    # OR
    very_sparse=True,         # Sparsity = sqrt(min_dim)
)

model = get_peft_model(model, config)
# Faster matmuls with sparse matrices
</syntaxhighlight>

=== Multiple RandLoRA Adapters ===
<syntaxhighlight lang="python">
# All adapters must use same projection_prng_key
config1 = RandLoraConfig(
    r=16,
    projection_prng_key=42,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, config1)
model.add_adapter("task2", config1)  # Same key required
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
