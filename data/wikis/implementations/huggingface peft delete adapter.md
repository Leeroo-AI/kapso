{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Adapter]], [[domain::Memory_Management]], [[domain::Multi_Task]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for removing an adapter from a PeftModel to free memory.

=== Description ===

`delete_adapter` permanently removes an adapter from the model, freeing the associated GPU memory. This is useful when you no longer need a specific adapter and want to reclaim VRAM for other adapters or operations.

=== Usage ===

Use this to remove adapters you no longer need. You cannot delete the currently active adapter—switch to a different adapter first. After deletion, the adapter name can be reused.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/peft_model.py
* '''Lines:''' L1083-1101

=== Signature ===
<syntaxhighlight lang="python">
def delete_adapter(self, adapter_name: str) -> None:
    """
    Delete an adapter from the model.

    Args:
        adapter_name: Name of adapter to remove.

    Raises:
        ValueError: If adapter doesn't exist or is currently active.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Method on PeftModel
# model.delete_adapter("adapter_name")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| adapter_name || str || Yes || Name of adapter to delete
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || - || Adapter removed, memory freed
|}

== Usage Examples ==

=== Delete Unused Adapter ===
<syntaxhighlight lang="python">
# Load multiple adapters
model.load_adapter("adapter1", adapter_name="task1")
model.load_adapter("adapter2", adapter_name="task2")

# Switch away from adapter to delete
model.set_adapter("task2")

# Delete unused adapter
model.delete_adapter("task1")

# Memory freed, task1 no longer available
print(list(model.peft_config.keys()))  # ["task2"]
</syntaxhighlight>

=== Cleanup After Merging ===
<syntaxhighlight lang="python">
# After merging adapters, delete originals
model.add_weighted_adapter(
    ["math", "code"],
    [0.5, 0.5],
    "merged",
    combination_type="linear"
)

model.set_adapter("merged")

# Clean up original adapters
model.delete_adapter("math")
model.delete_adapter("code")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Adapter_Deletion]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
