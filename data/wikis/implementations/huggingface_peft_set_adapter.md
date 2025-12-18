{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft/conceptual_guides/lora#manage-multiple-adapters]]
|-
! Domains
| [[domain::Adapter]], [[domain::Multi_Task]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for switching the active adapter in a multi-adapter PeftModel.

=== Description ===

`set_adapter` activates a specific adapter (or combination of adapters) for inference or training. Only the active adapter affects model outputs. This enables rapid switching between task-specific adapters without reloading models.

=== Usage ===

Use this after loading multiple adapters to select which one should be active. Pass a single adapter name for exclusive activation, or a list to combine multiple adapters (outputs summed).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/peft_model.py
* '''Lines:''' L1477-1504

=== Signature ===
<syntaxhighlight lang="python">
def set_adapter(self, adapter_name: Union[str, list[str]]) -> None:
    """
    Set the active adapter(s).

    Args:
        adapter_name: Single adapter name or list of adapters to activate.
                     If list, their outputs are summed.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Method on PeftModel
# model.set_adapter("adapter_name")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| adapter_name || str or list[str] || Yes || Adapter name(s) to activate
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || - || Model state updated in-place
|}

== Usage Examples ==

=== Switch Between Adapters ===
<syntaxhighlight lang="python">
# Load multiple adapters
model.load_adapter("math-adapter", adapter_name="math")
model.load_adapter("code-adapter", adapter_name="code")

# Use math adapter
model.set_adapter("math")
math_output = model.generate(**math_prompt)

# Switch to code adapter
model.set_adapter("code")
code_output = model.generate(**code_prompt)
</syntaxhighlight>

=== Combine Multiple Adapters ===
<syntaxhighlight lang="python">
# Activate multiple adapters (outputs summed)
model.set_adapter(["math", "code"])
combined_output = model.generate(**inputs)
</syntaxhighlight>

=== Check Active Adapter ===
<syntaxhighlight lang="python">
# Check current adapter
print(model.active_adapter)  # "math"

# Check all available
print(list(model.peft_config.keys()))  # ["math", "code"]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Adapter_Switching]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
