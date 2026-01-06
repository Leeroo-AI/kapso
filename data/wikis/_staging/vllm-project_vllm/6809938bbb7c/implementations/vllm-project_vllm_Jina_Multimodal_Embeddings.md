{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Multimodal]], [[domain::Embeddings]], [[domain::Vision]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Offline example demonstrating multimodal embeddings with Jina v4 models supporting both text and image inputs.

=== Description ===
This example showcases Jina Embeddings v4's multimodal capabilities using vLLM's token_embed pooling task. It demonstrates encoding both multilingual text (German and Japanese) and images into a shared embedding space. The example includes custom embedding pooling logic that handles vision tokens specially, extracting embeddings between vision start/end markers and applying proper normalization. This enables cross-modal retrieval where text queries can match image content and vice versa.

=== Usage ===
Use this example for multimodal search applications, cross-lingual retrieval systems, or any scenario requiring embeddings that work across text and images. It's particularly valuable for building search systems that handle both documents and visual content in multiple languages.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/token_embed/jina_embeddings_v4.py examples/pooling/token_embed/jina_embeddings_v4.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run Jina multimodal embedding example
python jina_embeddings_v4.py
</syntaxhighlight>

== Key Concepts ==

=== Multimodal Embedding Space ===
Jina v4 models embed text and images into the same vector space, enabling cross-modal similarity comparisons and retrieval.

=== Vision Token Extraction ===
For image inputs, the example identifies vision tokens between VISION_START_TOKEN_ID (151652) and VISION_END_TOKEN_ID (151653), pooling only those tokens for image embeddings.

=== Text Token Pooling ===
For text-only inputs (no vision tokens), all output tokens are used for embedding computation.

=== Mean Pooling and Normalization ===
Embeddings are created by averaging token embeddings (mean pooling) and then L2-normalizing the result for cosine similarity comparisons.

=== Multilingual Support ===
The example demonstrates embedding text in German ("Ein wunderschöner Sonnenuntergang am Strand") and Japanese ("浜辺に沈む美しい夕日"), showing the model's multilingual capabilities.

=== Structured Prompts ===
Uses TextPrompt objects with multi_modal_data parameter to pass image data alongside text prompts in a structured format.

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from vllm import LLM
from vllm.inputs.data import TextPrompt
from vllm.multimodal.utils import fetch_image

# Initialize model
model = LLM(
    model="jinaai/jina-embeddings-v4-vllm-text-matching",
    runner="pooling",
    max_model_len=1024,
    gpu_memory_utilization=0.8
)

# Create text prompts (multilingual)
text1_prompt = TextPrompt(
    prompt="Query: Ein wunderschöner Sonnenuntergang am Strand"  # German
)
text2_prompt = TextPrompt(
    prompt="Query: 浜辺に沈む美しい夕日"  # Japanese
)

# Create image prompt
image = fetch_image(
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/eskimo.jpg"
)
image_prompt = TextPrompt(
    prompt="<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n",
    multi_modal_data={"image": image}
)

# Encode all prompts
prompts = [text1_prompt, text2_prompt, image_prompt]
outputs = model.encode(prompts, pooling_task="token_embed")

# Custom pooling for vision vs text tokens
VISION_START_TOKEN_ID, VISION_END_TOKEN_ID = 151652, 151653

embeddings = []
for output in outputs:
    if VISION_START_TOKEN_ID in output.prompt_token_ids:
        # Extract vision tokens only
        img_start = torch.where(
            torch.tensor(output.prompt_token_ids) == VISION_START_TOKEN_ID
        )[0][0]
        img_end = torch.where(
            torch.tensor(output.prompt_token_ids) == VISION_END_TOKEN_ID
        )[0][0]
        embeds_tensor = output.outputs.data[img_start:img_end+1]
    else:
        # Use all tokens for text
        embeds_tensor = output.outputs.data

    # Mean pool and normalize
    pooled = embeds_tensor.sum(dim=0, dtype=torch.float32) / embeds_tensor.shape[0]
    normalized = torch.nn.functional.normalize(pooled, dim=-1)
    embeddings.append(normalized)

# All embeddings are now in the same space
for i, emb in enumerate(embeddings):
    print(f"Embedding {i} shape: {emb.shape}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[related::Implementation:vllm-project_vllm_Multi_Vector_Embeddings]]
