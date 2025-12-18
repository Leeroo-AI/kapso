{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]], [[domain::Transfer_Learning]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for loading pre-trained transformer models as the foundation for parameter-efficient fine-tuning.

=== Description ===

Base Model Loading is the first step in any PEFT workflow. It involves loading a pre-trained transformer model that will serve as the frozen foundation for adapter training. The principle emphasizes that the base model's weights remain unchanged during fine-tuning—only the adapter layers are trained. This enables efficient training while preserving the model's pre-trained knowledge.

The key insight is that modern LLMs contain general knowledge that should be preserved, while adapters learn task-specific modifications. Proper loading includes selecting appropriate precision (float16/bfloat16) and device placement strategies.

=== Usage ===

Apply this principle at the start of any LoRA/PEFT fine-tuning workflow. Consider:
* **Model selection:** Choose a base model appropriate for your task (decoder-only for generation, encoder-decoder for seq2seq)
* **Precision:** Use float16 or bfloat16 for memory efficiency on consumer GPUs
* **Device mapping:** Use "auto" for multi-GPU setups, explicit GPU ID for single GPU
* **Attention implementation:** Flash Attention 2 for speed, SDPA for compatibility

== Theoretical Basis ==

The foundational principle behind PEFT is that pre-trained models form a "good starting point" in parameter space. The base model represents:

'''Frozen Knowledge:'''
<math>W_0 \in \mathbb{R}^{d \times k}</math>

Where <math>W_0</math> contains pre-trained weights that encode:
- Language understanding (syntax, semantics)
- World knowledge from pre-training corpus
- Task-general representations

'''Adaptation Hypothesis:'''

LoRA hypothesizes that task-specific adaptation can be achieved through low-rank updates:
<math>W = W_0 + \Delta W</math>

Where <math>\Delta W = BA</math> with <math>B \in \mathbb{R}^{d \times r}</math> and <math>A \in \mathbb{R}^{r \times k}</math>, and <math>r \ll min(d, k)</math>.

This only works if <math>W_0</math> is properly initialized—hence the importance of correct base model loading.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_AutoModelForCausalLM_from_pretrained]]
