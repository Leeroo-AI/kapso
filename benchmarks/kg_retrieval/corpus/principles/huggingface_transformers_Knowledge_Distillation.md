# Principle: Knowledge_Distillation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Distilling the Knowledge in a Neural Network|https://arxiv.org/abs/1503.02531]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Compression]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Training a compact student model to match the softened output distribution of a larger teacher, usually with a temperature-scaled divergence loss added to or replacing the label loss.

=== Description ===

A teacher's full probability vector contains information about similarities among classes that a hard label hides. Raising the softmax temperature exposes it, and a student trained to reproduce those softened outputs generalises better than one trained on labels alone.

=== Usage ===

Use it when a smaller architecture must be deployed and a strong model, or its outputs, is available.

== Theoretical Basis ==

With logits z_t and z_s and temperature T, the loss is T^2 * KL(softmax(z_t / T) || softmax(z_s / T)), optionally mixed with cross entropy on true labels. The T^2 factor keeps gradient magnitudes comparable across temperatures.

== Related Pages ==

* [[implemented_by::Implementation:huggingface_transformers_Distillation_Loss_KL]]
