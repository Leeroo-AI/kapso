{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Visual Instruction Tuning (LLaVA)|https://arxiv.org/abs/2304.08485]]
* [[source::Paper|Qwen-VL|https://arxiv.org/abs/2308.12966]]
* [[source::Doc|Vision Fine-tuning Guide|https://docs.unsloth.ai/basics/vision-fine-tuning]]
|-
! Domains
| [[domain::Vision_Language_Models]], [[domain::Deep_Learning]], [[domain::Multimodal]], [[domain::Transfer_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Technique for loading and configuring Vision-Language Models (VLMs) with memory-efficient quantization while preserving multimodal understanding capabilities.

=== Description ===

Vision-Language Model loading in Unsloth handles the unique challenges of multimodal architectures:

**Architecture Types:**
1. **Cross-attention VLMs** (e.g., Flamingo): Vision features attend to language via cross-attention layers
2. **Unified VLMs** (e.g., Qwen-VL, LLaVA): Vision tokens are projected into the same embedding space as text

**Loading Challenges:**
- Vision encoders (ViT, SigLIP) have different quantization requirements than language models
- Image tokens expand sequence length significantly (~1000+ tokens per image)
- Different models use different image token placeholders and formats

**Unsloth Optimizations:**
- Automatic architecture detection from config
- Selective quantization (language model in 4-bit, vision encoder often in 16-bit)
- Efficient attention for long sequences with image tokens
- Memory-efficient multi-image processing

This enables training 7B VLMs on 16GB GPUs and 70B VLMs on 48GB GPUs.

=== Usage ===

Use VLM loading when:
* Fine-tuning for visual question answering (VQA)
* Training document understanding / OCR models
* Building image captioning systems
* Creating multi-image reasoning models

Requirements:
- GPU with 16GB+ VRAM (24GB+ recommended)
- CUDA capability 8.0+ for best performance
- transformers >= 4.49.0 for latest VLM architectures

== Theoretical Basis ==

=== VLM Architecture Patterns ===

<syntaxhighlight lang="python">
# Abstract VLM architecture
class VisionLanguageModel:
    def __init__(self):
        # Vision encoder (typically frozen or lightly tuned)
        self.vision_encoder = SigLIP()  # or ViT, CLIP, etc.

        # Projection layer (trainable)
        self.projector = MLP(vision_dim=1024, text_dim=4096)

        # Language model (main training target)
        self.language_model = LlamaForCausalLM()

    def forward(self, images, text_ids):
        # 1. Encode images
        image_features = self.vision_encoder(images)

        # 2. Project to text embedding space
        image_embeds = self.projector(image_features)

        # 3. Interleave with text embeddings
        text_embeds = self.embed_tokens(text_ids)
        combined = interleave(image_embeds, text_embeds)

        # 4. Forward through language model
        return self.language_model(inputs_embeds=combined)
</syntaxhighlight>

=== Image Token Handling ===

VLMs replace image placeholder tokens with projected image features:

<syntaxhighlight lang="python">
# Abstract image token replacement
def process_multimodal_input(text, images, tokenizer, vision_encoder):
    # Tokenize text with image placeholders
    # e.g., "<image>What is this?" -> [IMG_TOKEN, "What", "is", "this", "?"]
    input_ids = tokenizer(text)

    # Find image token positions
    image_positions = find_token(input_ids, IMG_TOKEN)

    # Encode images to features
    image_features = vision_encoder(images)  # [num_images, num_patches, dim]

    # Replace single token with sequence of patch features
    # One image becomes ~256-1024 tokens depending on resolution
    expanded_embeds = expand_image_tokens(input_ids, image_features, image_positions)

    return expanded_embeds
</syntaxhighlight>

=== Selective Quantization Strategy ===

Different components have different quantization tolerance:

{| class="wikitable"
|-
! Component !! Recommended Precision !! Reason
|-
| Vision Encoder || float16 / bfloat16 || Sensitive to quantization; spatial features degrade
|-
| Projector || float16 || Small layer, minimal memory savings from quant
|-
| Language Model || 4-bit NF4 || Large component, well-studied quantization
|-
| LM Head || float16 || Output quality sensitive layer
|}

<syntaxhighlight lang="python">
# Abstract selective quantization
def load_vlm_with_selective_quant(model_name):
    config = AutoConfig.from_pretrained(model_name)

    # Configure quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=[
            "vision_tower",      # Keep vision in fp16
            "vision_projection", # Keep projector in fp16
            "lm_head",          # Keep output in fp16
        ]
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=quant_config,
    )

    return model
</syntaxhighlight>

=== Multi-Image Attention ===

For models supporting multiple images, attention patterns must handle:

<syntaxhighlight lang="python">
# Abstract multi-image attention
def multi_image_attention(query, key, value, image_positions):
    # Standard causal attention for text
    # But image tokens can attend to each other within same image

    # Build attention mask
    mask = causal_mask(seq_len)

    # Allow intra-image attention
    for img_start, img_end in image_positions:
        # Image tokens can see all tokens in same image
        mask[img_start:img_end, img_start:img_end] = 0

    return scaled_dot_product_attention(query, key, value, mask)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Optimization]]
