{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft/quicktour#load-and-use-a-peft-model]]
|-
! Domains
| [[domain::Adapter]], [[domain::Model_Loading]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for loading a trained PEFT adapter onto a base model for inference or further training.

=== Description ===

`PeftModel.from_pretrained` is the primary method for loading pre-trained PEFT adapters. It loads the adapter configuration and weights from disk or HuggingFace Hub, injects adapter layers into the provided base model, and restores the trained adapter weights. This enables inference with task-specific adapters.

=== Usage ===

Use this when you have a trained adapter saved and want to use it for inference or continue training. The base model must match the model used during training. For inference, set `is_trainable=False` (default) to freeze adapters.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/peft_model.py
* '''Lines:''' L388-604

=== Signature ===
<syntaxhighlight lang="python">
@classmethod
def from_pretrained(
    cls,
    model: torch.nn.Module,
    model_id: Union[str, os.PathLike],
    adapter_name: str = "default",
    is_trainable: bool = False,
    config: Optional[PeftConfig] = None,
    autocast_adapter_dtype: bool = True,
    ephemeral_gpu_offload: bool = False,
    low_cpu_mem_usage: bool = False,
    **kwargs: Any,
) -> PeftModel:
    """
    Load a PEFT adapter from a pretrained checkpoint.

    Args:
        model: Base model to adapt
        model_id: Adapter path (local or HuggingFace Hub)
        adapter_name: Name for the adapter. Default: "default"
        is_trainable: Load for training (True) or inference (False)
        config: Override config (usually auto-loaded)
        autocast_adapter_dtype: Cast adapter weights for stability

    Returns:
        PeftModel with loaded adapter
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import PeftModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Base model matching the adapter's base_model_name_or_path
|-
| model_id || str || Yes || HuggingFace Hub ID or local path to adapter
|-
| adapter_name || str || No || Name for loaded adapter. Default: "default"
|-
| is_trainable || bool || No || False for inference, True for training. Default: False
|-
| autocast_adapter_dtype || bool || No || Cast float16/bfloat16 to float32. Default: True
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| peft_model || PeftModel || Model with loaded adapter ready for inference/training
|}

== Usage Examples ==

=== Load for Inference ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "username/my-lora-adapter",  # HuggingFace Hub path
    is_trainable=False,          # Inference mode
)

# 3. Use for generation
model.eval()
outputs = model.generate(**inputs)
</syntaxhighlight>

=== Load from Local Path ===
<syntaxhighlight lang="python">
from peft import PeftModel

# Load from local directory
model = PeftModel.from_pretrained(
    base_model,
    "./trained-adapter",
    adapter_name="math_adapter",
)
</syntaxhighlight>

=== Continue Training ===
<syntaxhighlight lang="python">
from peft import PeftModel

# Load for further training
model = PeftModel.from_pretrained(
    base_model,
    "./checkpoint-1000",
    is_trainable=True,  # Enable gradients
)

# Continue training
trainer.train(resume_from_checkpoint=True)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Adapter_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
