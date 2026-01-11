# Principle: Vision_Training_Mode

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PyTorch Training|https://pytorch.org/docs/stable/notes/autograd.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Configuration of Vision-Language Models for gradient-enabled training mode with memory-efficient checkpointing.

=== Description ===

Vision Training Mode ensures the model is properly configured for fine-tuning:
* **Gradient computation** enabled for trainable parameters
* **Gradient checkpointing** activated for memory efficiency
* **Training mode** set (affects dropout, batch norm)

VLMs require explicit mode switching because inference and training have different requirements.

=== Usage ===

Set training mode before starting the training loop. Unsloth typically handles this automatically, but explicit calls may be needed for custom training loops.

== Theoretical Basis ==

=== Training vs Inference Mode ===

| Aspect | Training Mode | Inference Mode |
|--------|---------------|----------------|
| Gradients | Enabled | Disabled |
| Checkpointing | Active | Disabled |
| Memory | Higher | Lower |
| Speed | Slower | Faster |
| Dropout | Active | Disabled |

=== Gradient Checkpointing ===

Trades compute for memory:

<math>
\text{Memory}_{checkpoint} = O(\sqrt{L}) \text{ vs } O(L)
</math>

Activations are recomputed during backward pass instead of stored.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_FastBaseModel_for_training]]
