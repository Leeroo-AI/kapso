{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Visual Instruction Tuning|https://arxiv.org/abs/2304.08485]]
* [[source::Paper|vLLM: Easy, Fast, and Cheap LLM Serving|https://arxiv.org/abs/2309.06180]]
|-
! Domains
| [[domain::Vision]], [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of generating text outputs from combined vision and language inputs through a unified multimodal model architecture.

=== Description ===

Multimodal Generation combines image understanding with text generation. The process:

1. **Vision Encoding:** Images processed through vision encoder (CLIP, SigLIP)
2. **Projection:** Vision features mapped to language model space
3. **Embedding Combination:** Image and text embeddings interleaved
4. **Autoregressive Generation:** Language model generates text response
5. **Output Processing:** Generated tokens decoded to text

This enables capabilities like image captioning, visual QA, and document understanding.

=== Usage ===

Apply multimodal generation when:
- Building image captioning systems
- Creating visual question answering applications
- Analyzing documents with images and text
- Implementing multimodal chatbots

== Theoretical Basis ==

'''Multimodal Architecture:'''

<syntaxhighlight lang="text">
     ┌─────────────┐
     │   Image     │
     └──────┬──────┘
            │
    ┌───────▼───────┐
    │ Vision Encoder│ (CLIP ViT)
    │   (frozen)    │
    └───────┬───────┘
            │ [patch embeddings]
    ┌───────▼───────┐
    │  Projection   │ (MLP / QFormer)
    │    Layer      │
    └───────┬───────┘
            │ [projected embeddings]
            │
            ▼
    ┌─────────────────────────────┐
    │  [img][img]...[img][text]   │ Combined sequence
    └─────────────┬───────────────┘
                  │
    ┌─────────────▼───────────────┐
    │   Language Model (LLaMA)    │
    │     (fine-tuned)            │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │    Generated Text           │
    └─────────────────────────────┘
</syntaxhighlight>

'''Token Sequence Construction:'''

<syntaxhighlight lang="python">
# Conceptual multimodal sequence
def build_multimodal_sequence(text_tokens, image_embeddings, placeholder_positions):
    """
    Builds final embedding sequence with images inserted.

    Args:
        text_tokens: Tokenized text with placeholder tokens
        image_embeddings: Encoded image features [num_images, num_patches, dim]
        placeholder_positions: Where to insert image embeddings

    Returns:
        Combined embedding sequence
    """
    sequence = []
    img_idx = 0

    for i, token in enumerate(text_tokens):
        if i in placeholder_positions:
            # Insert image embeddings
            sequence.extend(image_embeddings[img_idx])
            img_idx += 1
        else:
            # Regular text token
            sequence.append(embed_token(token))

    return torch.stack(sequence)
</syntaxhighlight>

'''Generation Flow:'''

<syntaxhighlight lang="python">
# Multimodal generation (conceptual)
def generate_multimodal(prompt_dict, sampling_params):
    prompt = prompt_dict["prompt"]
    images = prompt_dict["multi_modal_data"]["image"]

    # 1. Encode images
    image_features = vision_encoder(images)
    image_embeddings = projection(image_features)

    # 2. Tokenize text
    text_tokens = tokenizer.encode(prompt)

    # 3. Find placeholder positions
    placeholder_pos = find_placeholders(text_tokens)

    # 4. Build combined sequence
    input_embeds = build_multimodal_sequence(
        text_tokens, image_embeddings, placeholder_pos
    )

    # 5. Run autoregressive generation
    output_tokens = language_model.generate(
        inputs_embeds=input_embeds,
        **sampling_params,
    )

    # 6. Decode to text
    return tokenizer.decode(output_tokens)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LLM_generate_mm]]
