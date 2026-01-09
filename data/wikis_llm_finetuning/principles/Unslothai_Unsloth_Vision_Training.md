# Principle: Vision_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LLaVA|https://arxiv.org/abs/2304.08485]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Training]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Training methodology for Vision-Language Models using supervised fine-tuning on image-text pairs with specialized data collation.

=== Description ===

Vision Training adapts VLMs to specific tasks through supervised learning on multimodal data. Key differences from text-only training:

1. **Data collation**: Must handle variable-size images and batch them
2. **Processing**: AutoProcessor for both images and text
3. **Memory**: Higher due to image embeddings
4. **Loss masking**: Only on text responses, not visual tokens

=== Usage ===

Apply Vision Training when:
* Fine-tuning VLMs for specific visual tasks
* Adapting to new image domains
* Teaching new output formats for visual inputs

== Theoretical Basis ==

=== VLM Loss Function ===

Loss computed on text response tokens:

<math>
\mathcal{L} = -\sum_{t \in \text{text}} \log P(y_t | \text{images}, \text{prompt}, y_{<t})
</math>

Visual tokens serve as context but are not predicted.

=== Data Collation ===

UnslothVisionDataCollator handles:
* Padding images to consistent size
* Creating attention masks for variable lengths
* Batching pixel values efficiently
* Preserving image-text alignment

=== Memory Considerations ===

VLM training uses more memory due to:
* Image tensors (pixel values)
* Vision encoder activations
* Cross-modal attention patterns

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_SFTTrainer_vision]]
