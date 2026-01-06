{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Visual Instruction Tuning|https://arxiv.org/abs/2304.08485]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Vision]], [[domain::NLP]], [[domain::Prompt_Engineering]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of constructing prompts that combine text instructions with image placeholder tokens for vision-language model inference.

=== Description ===

VLM Prompt Construction involves creating input sequences that interleave text tokens with image embedding positions. Key challenges:

1. **Placeholder Placement:** Correct position for image embeddings
2. **Model Compatibility:** Different models use different tokens
3. **Chat Templates:** Proper formatting for instruction-tuned VLMs
4. **Multi-Image Ordering:** Matching placeholders to image data
5. **Token Counting:** Accounting for image token expansion

=== Usage ===

Construct VLM prompts when:
- Building visual question answering systems
- Creating image comparison applications
- Designing document understanding pipelines
- Implementing multimodal chatbots

== Theoretical Basis ==

'''Prompt Structure:'''

<syntaxhighlight lang="text">
[System Prompt] + [Image Placeholder(s)] + [User Question] + [Response Trigger]

Example (LLaVA):
"USER: <image>\nWhat is shown in this image?\nASSISTANT:"
       ↑ Image embeddings inserted here
</syntaxhighlight>

'''Placeholder Replacement:'''

During inference, placeholders are replaced with image embeddings:

<syntaxhighlight lang="python">
# Conceptual embedding insertion
def process_vlm_prompt(prompt, images, tokenizer, vision_encoder):
    # Tokenize text
    tokens = tokenizer.encode(prompt)

    # Find placeholder positions
    placeholder_positions = find_placeholder_tokens(tokens)

    # Encode images
    image_embeddings = [vision_encoder(img) for img in images]

    # Insert embeddings at placeholder positions
    final_embeddings = insert_at_positions(
        text_embeddings=embed(tokens),
        image_embeddings=image_embeddings,
        positions=placeholder_positions,
    )

    return final_embeddings
</syntaxhighlight>

'''Model-Specific Formats:'''

| Model | Placeholder | Example |
|-------|-------------|---------|
| LLaVA | `<image>` | `"<image>\nDescribe this."` |
| Qwen-VL | `<img>url</img>` | `"<img>http://...</img>\nWhat is this?"` |
| Phi-3-Vision | `<\|image_N\|>` | `"<\|image_1\|>\nAnalyze."` |
| Pixtral | `[IMG]` | `"[IMG]\nCaption this image."` |

'''Chat Template Usage:'''

<syntaxhighlight lang="python">
# Preferred: Use chat template for automatic formatting
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is this?"},
        ],
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
# Returns properly formatted prompt for specific model
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_VLM_prompt_format]]
