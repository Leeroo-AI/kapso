{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft/conceptual_guides/lora#manage-multiple-adapters]]
|-
! Domains
| [[domain::Adapter]], [[domain::Multi_Task]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for loading additional adapters into an existing PeftModel for multi-adapter scenarios.

=== Description ===

`load_adapter` loads an additional adapter into a model that already has PEFT enabled. Unlike `from_pretrained`, this adds to an existing PeftModel rather than creating a new one. This enables multi-adapter inference where you can switch between different task-specific adapters.

=== Usage ===

Use this after creating a PeftModel to add additional adapters. Each adapter needs a unique name. The newly loaded adapter is not automatically activated—use `set_adapter()` to switch to it.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/peft_model.py
* '''Lines:''' L1309-1475

=== Signature ===
<syntaxhighlight lang="python">
def load_adapter(
    self,
    model_id: Union[str, os.PathLike],
    adapter_name: str,
    is_trainable: bool = False,
    torch_device: Optional[str] = None,
    autocast_adapter_dtype: bool = True,
    ephemeral_gpu_offload: bool = False,
    low_cpu_mem_usage: bool = False,
    **kwargs: Any,
):
    """
    Load an additional adapter into the model.

    Args:
        model_id: Adapter path (local or HuggingFace Hub)
        adapter_name: Unique name for this adapter
        is_trainable: Enable gradients for training. Default: False
        torch_device: Target device. Default: auto-infer
        autocast_adapter_dtype: Cast weights for stability

    Returns:
        Load result with missing/unexpected keys
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Method on PeftModel
# model.load_adapter(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_id || str || Yes || Adapter path (local or Hub)
|-
| adapter_name || str || Yes || Unique name for this adapter
|-
| is_trainable || bool || No || Load with gradients enabled. Default: False
|-
| torch_device || str || No || Target device. Default: auto
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| load_result || LoadResult || Object with missing_keys, unexpected_keys lists
|}

== Usage Examples ==

=== Load Multiple Task Adapters ===
<syntaxhighlight lang="python">
from peft import PeftModel

# Start with one adapter
model = PeftModel.from_pretrained(
    base_model,
    "username/general-adapter",
    adapter_name="general",
)

# Add task-specific adapters
model.load_adapter(
    "username/math-adapter",
    adapter_name="math",
)
model.load_adapter(
    "username/code-adapter",
    adapter_name="code",
)

# Switch between adapters
model.set_adapter("math")
math_output = model.generate(**inputs)

model.set_adapter("code")
code_output = model.generate(**inputs)
</syntaxhighlight>

=== Load Local Adapter ===
<syntaxhighlight lang="python">
# Load from local path
model.load_adapter(
    "./my-local-adapter",
    adapter_name="custom",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Multi_Adapter_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
