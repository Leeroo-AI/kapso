# Heuristic: Gradient_Accumulation_For_Small_Batches

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PyTorch Automatic Mixed Precision examples|https://pytorch.org/docs/stable/notes/amp_examples.html]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Sum gradients over several micro-batches before each optimizer step so the effective batch reaches its target even when memory allows just a handful of samples per forward pass.

=== Description ===

Gradients are linear in the loss, so adding the gradients of several small groups and stepping once equals one step on their union, up to normalisation effects.

=== Usage ===

Use it whenever the desired update size is larger than what one forward pass can hold.

== The Insight (Rule of Thumb) ==

Pick the target number of samples per update first, then divide it by the per-device micro-batch and the number of devices to get the number of accumulation steps. Divide each micro-batch loss by that number so the summed gradient has the right scale, and call the optimizer only on the last micro-batch.

== Reasoning ==

Batch-statistics layers see only the micro-batch, so their behaviour still differs from a true large group. Learning-rate schedules should count optimizer steps, not micro-batches.
