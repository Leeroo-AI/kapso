{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Adapter]], [[domain::Memory_Management]], [[domain::Resource_Cleanup]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for removing adapters from a model to reclaim GPU memory.

=== Description ===

Adapter Deletion permanently removes adapter weights from the model, freeing associated memory. This is essential for:
* Memory management with many adapters
* Cleaning up after merging operations
* Replacing adapters with new versions
* Resource optimization in production

=== Usage ===

Apply this when:
* An adapter is no longer needed
* Memory is constrained
* After merging adapters (delete originals)
* Cleaning up failed experiments

Constraint: Cannot delete the currently active adapter.

== Theoretical Basis ==

'''Memory Reclamation:'''

Deleting an adapter with rank r frees:
<math>\text{Memory freed} = (d \times r + r \times k) \times \text{sizeof(dtype)}</math>

For typical dimensions and r=16 in float16:
<math>\text{Memory} \approx 2 \times 16 \times (4096 + 4096) \times 2 \approx 0.5 \text{ MB/layer}</math>

'''Deletion Process:'''

<syntaxhighlight lang="python">
# Pseudo-code for adapter deletion
def delete_adapter(self, adapter_name):
    # Validate not active
    if adapter_name == self.active_adapter:
        raise ValueError("Cannot delete active adapter")

    # Remove from all layers
    for module in self.modules():
        if isinstance(module, LoraLayer):
            if adapter_name in module.lora_A:
                del module.lora_A[adapter_name]
                del module.lora_B[adapter_name]
                del module.scaling[adapter_name]

    # Remove config
    del self.peft_config[adapter_name]

    # Trigger garbage collection
    torch.cuda.empty_cache()
</syntaxhighlight>

'''Memory Management Best Practices:'''

<syntaxhighlight lang="python">
# Pattern: Clean up after merging
model.add_weighted_adapter(["a", "b"], [0.5, 0.5], "merged")
model.set_adapter("merged")

# Now safe to delete originals
model.delete_adapter("a")
model.delete_adapter("b")

# Force GPU memory cleanup
import gc
gc.collect()
torch.cuda.empty_cache()
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_delete_adapter]]
