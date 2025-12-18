{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Repo|Megatron-LM|https://github.com/NVIDIA/Megatron-LM]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Tensor_Parallelism]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

LoRA layer implementation for Megatron-LM tensor parallel linear layers, handling row and column parallelism with appropriate LoRA matrix splitting.

=== Description ===

LoraParallelLinear implements LoRA for Megatron-LM's tensor parallel layers. For RowParallelLinear (where inputs are split across devices), lora_A is implemented as a parallel row layer while lora_B remains a standard linear layer. For ColumnParallelLinear (where outputs are split), lora_A is standard while lora_B is a parallel column layer. This ensures input/output shapes remain consistent with the base parallel layer while adding low-rank adaptation.

=== Usage ===

Use LoraParallelLinear when fine-tuning models that use Megatron-LM tensor parallelism. The layer is automatically dispatched via `dispatch_megatron` when target layers are `RowParallelLinear` or `ColumnParallelLinear`. Requires Megatron configuration for parallel settings.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/tp_layer.py src/peft/tuners/lora/tp_layer.py]
* '''Lines:''' 1-351

=== Signature ===
<syntaxhighlight lang="python">
class LoraParallelLinear(nn.Module, LoraLayer):
    """
    LoRA for Megatron tensor parallel linear layers.

    For RowParallelLinear: lora_A is row-parallel, lora_B is standard
    For ColumnParallelLinear: lora_A is standard, lora_B is column-parallel
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        backend,  # megatron_core.tensor_parallel
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ):
        """Initialize with Megatron backend and config."""

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        init_method,
        input_is_parallel,
        gather_output,
        **parallel_linear_kwargs,
    ):
        """Create parallel LoRA layers based on base layer type."""

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Forward with parallel LoRA computation."""

def dispatch_megatron(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config,
    **kwargs,
) -> Optional[torch.nn.Module]:
    """Dispatch LoRA for Megatron parallel layers."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.lora.tp_layer import LoraParallelLinear, dispatch_megatron
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| base_layer || nn.Module || Yes || Megatron RowParallelLinear or ColumnParallelLinear
|-
| adapter_name || str || Yes || Name for the adapter
|-
| backend || module || Yes || megatron_core.tensor_parallel module
|-
| megatron_config || dict/TransformerConfig || Yes || Megatron configuration
|-
| r || int || Yes || LoRA rank
|-
| lora_alpha || int || No || Scaling factor
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward output || tuple[Tensor, Tensor] || (result, bias) matching Megatron signature
|-
| get_delta_weight() || torch.Tensor || LoRA delta weight B @ A * scaling
|}

== Usage Examples ==

=== LoRA with Megatron Configuration ===
<syntaxhighlight lang="python">
from peft import LoraConfig, get_peft_model

# Configure LoRA with Megatron settings
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"],
    megatron_config={
        "tensor_model_parallel_size": 4,
        "sequence_parallel": True,
        # ... other Megatron config
    },
    megatron_core="megatron.core",  # Import path
)

# Apply to Megatron model
model = get_peft_model(megatron_model, config)
</syntaxhighlight>

=== Manual Dispatch ===
<syntaxhighlight lang="python">
from peft.tuners.lora.tp_layer import dispatch_megatron

# Dispatch automatically checks layer type
new_module = dispatch_megatron(
    target=target_layer,
    adapter_name="default",
    lora_config=lora_config,
    megatron_config=megatron_config,
)

if new_module is not None:
    # Replace target with LoRA-wrapped module
    parent.target_attr = new_module
</syntaxhighlight>

=== Forward Pass Handling ===
<syntaxhighlight lang="python">
# LoraParallelLinear returns (result, bias) tuple
# matching Megatron's parallel layer signature
result, bias = lora_parallel_layer(x)

# For RowParallelLinear:
#   lora_A splits input across devices
#   lora_B gathers output
# For ColumnParallelLinear:
#   lora_A operates on full input
#   lora_B splits output across devices
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
