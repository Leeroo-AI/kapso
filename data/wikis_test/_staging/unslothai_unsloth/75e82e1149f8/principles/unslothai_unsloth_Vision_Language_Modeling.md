# Principle: Vision-Language Modeling

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LLaVA: Visual Instruction Tuning|https://arxiv.org/abs/2304.08485]]
* [[source::Paper|Qwen-VL: A Versatile Vision-Language Model|https://arxiv.org/abs/2308.12966]]
* [[source::Paper|CLIP: Learning Transferable Visual Models|https://arxiv.org/abs/2103.00020]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Multimodal]], [[domain::Vision_Language]], [[domain::Computer_Vision]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Multimodal architecture that combines vision encoders with large language models to enable understanding and generation of content involving both images and text through unified representation learning.

=== Description ===
Vision-Language Models (VLMs) extend LLMs to process visual inputs by integrating:

1. **Vision Encoder** - Extracts visual features from images (ViT, SigLIP, etc.)
2. **Projection Module** - Aligns visual features to text embedding space
3. **Language Model** - Processes combined visual-text representations
4. **Cross-Modal Attention** - Enables interaction between modalities

'''Architecture Variants:'''
- **Llama 3.2 Vision:** Cross-attention adapter between vision and language
- **Qwen2-VL:** Direct embedding concatenation with position encoding
- **Pixtral:** Native multimodal tokens interleaved with text

'''Capabilities:'''
- Image captioning and description
- Visual question answering (VQA)
- OCR and document understanding
- Multi-image reasoning
- Image-conditioned generation

=== Usage ===
Use Vision-Language fine-tuning when:
- Building OCR or document processing systems
- Creating visual assistants for specific domains
- Adapting VLMs for custom image understanding tasks
- Need grounded text generation from visual context

'''LoRA Configuration for VLMs:'''
- **finetune_vision_layers:** Enable for visual domain adaptation
- **finetune_language_layers:** Enable for task-specific responses
- Both: Comprehensive adaptation for new domains

== Theoretical Basis ==
'''VLM Architecture:'''

<syntaxhighlight lang="python">
class VisionLanguageModel:
    """Conceptual VLM architecture."""

    def __init__(self):
        self.vision_encoder = ViTEncoder()      # Image → patches → features
        self.projector = MLPProjection()        # Align vision to text space
        self.language_model = TransformerLM()   # Process unified sequence

    def forward(self, text_tokens, images):
        # 1. Extract visual features
        # Input: [B, C, H, W] images
        # Output: [B, num_patches, vision_dim]
        vision_features = self.vision_encoder(images)

        # 2. Project to language model dimension
        # Output: [B, num_patches, hidden_dim]
        visual_tokens = self.projector(vision_features)

        # 3. Embed text tokens
        # Output: [B, seq_len, hidden_dim]
        text_embeddings = self.language_model.embed(text_tokens)

        # 4. Concatenate visual and text tokens
        # The exact interleaving depends on architecture
        combined = interleave(visual_tokens, text_embeddings, positions)

        # 5. Process through transformer
        output = self.language_model(combined)

        return output
</syntaxhighlight>

'''Visual Token Generation:'''
<math>
V_{tokens} = MLP(ViT(I)) \in \mathbb{R}^{N \times D}
</math>

Where:
- I is the input image
- N is the number of visual tokens (e.g., 256 for 16×16 patches)
- D is the language model hidden dimension

'''Multimodal Data Format:'''
<syntaxhighlight lang="python">
# OpenAI-style message format for VLMs
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "image": pil_image_or_url},
        {"type": "text", "text": "Describe it in detail."},
    ]
}

# Processing pipeline:
def process_multimodal_input(message, processor):
    """Convert multimodal message to model inputs."""
    text_parts = []
    images = []

    for content in message["content"]:
        if content["type"] == "text":
            text_parts.append(content["text"])
        elif content["type"] == "image":
            images.append(load_image(content["image"]))
            text_parts.append("<image>")  # Placeholder token

    # Apply processor (handles both text and images)
    inputs = processor(
        text=" ".join(text_parts),
        images=images,
        return_tensors="pt"
    )

    return inputs  # Contains input_ids, pixel_values, etc.
</syntaxhighlight>

'''Training Considerations:'''
- **Resolution Trade-off:** Higher resolution = more patches = more memory
- **Vision Layers:** Often beneficial to freeze early vision layers
- **Learning Rate:** Vision encoder typically uses lower learning rate

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]
* [[implemented_by::Implementation:unslothai_unsloth_UnslothVisionDataCollator]]

=== Tips and Tricks ===
