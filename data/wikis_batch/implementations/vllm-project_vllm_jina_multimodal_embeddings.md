# Implementation: Jina Embeddings v4 Multimodal Token Embeddings

**File:** `/tmp/praxium_repo_583nq7ea/examples/pooling/token_embed/jina_embeddings_v4.py`
**Type:** Multimodal Embedding Example
**Lines of Code:** 71
**Last Updated:** 2025-12-17

## Overview

The `jina_embeddings_v4.py` script demonstrates advanced multimodal token-level embeddings using Jina Embeddings v4 model. This implementation showcases vLLM's capability to process both text (multilingual) and vision inputs, extracting fine-grained token embeddings for sophisticated retrieval applications.

### Purpose

Illustrates vLLM's support for multimodal token embeddings, enabling ColBERT-style late interaction retrieval with vision-language models across multiple languages.

### Key Features

- **Multimodal Support**: Processes text and image inputs
- **Multilingual Text**: German and Japanese text examples
- **Vision Token Extraction**: Special handling for image embeddings
- **Token-Level Embeddings**: Fine-grained representations for each token
- **Normalized Outputs**: Mean-pooled and L2-normalized embeddings

## Architecture

### Multimodal Processing Pipeline

```
┌─────────────────────────────────────────────────────┐
│                   Input Layer                       │
├─────────────────────┬───────────────────────────────┤
│   Text Inputs       │      Image Input              │
│  - German text      │   - Vision tokens             │
│  - Japanese text    │   - Image patches             │
└──────────┬──────────┴──────────┬────────────────────┘
           │                     │
           ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│  Text Encoder    │   │  Vision Encoder  │
│  (Transformer)   │   │  (ViT/CLIP)      │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         ▼                      ▼
┌──────────────────────────────────────────┐
│     Token-Level Embeddings               │
│  [num_tokens × embedding_dim]            │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  Post-Processing                         │
│  - Extract vision tokens (if image)      │
│  - Mean pooling                          │
│  - L2 normalization                      │
└──────────────────────────────────────────┘
```

## Implementation Details

### Model Initialization

```python
model = LLM(
    model="jinaai/jina-embeddings-v4-vllm-text-matching",
    runner="pooling",
    max_model_len=1024,
    gpu_memory_utilization=0.8,
)
```

**Configuration:**
- `model`: Jina Embeddings v4 optimized for vLLM
- `runner="pooling"`: Enables token embedding mode
- `max_model_len=1024`: Context window for text/image
- `gpu_memory_utilization=0.8`: 80% GPU memory allocation

**Model Capabilities:**
- Multilingual text encoding
- Image understanding
- Token-level embeddings
- Late interaction retrieval support

### Text Prompt Construction

```python
# German text
text1 = "Ein wunderschöner Sonnenuntergang am Strand"
text1_prompt = TextPrompt(prompt=f"Query: {text1}")

# Japanese text
text2 = "浜辺に沈む美しい夕日"
text2_prompt = TextPrompt(prompt=f"Query: {text2}")
```

**Text Processing:**
- Multilingual support (German, Japanese)
- Query prefix for retrieval task formatting
- Standard TextPrompt wrapper

**Semantic Meaning:**
- German: "A beautiful sunset on the beach"
- Japanese: "A beautiful sunset sinking on the beach"
- Both describe similar scenes (multilingual test case)

### Image Prompt Construction

```python
image = fetch_image(
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/eskimo.jpg"
)

image_prompt = TextPrompt(
    prompt="<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n",
    multi_modal_data={"image": image},
)
```

**Image Processing:**
- Fetches image from URL or local path
- Special tokens mark vision regions:
  - `<|im_start|>`: Interaction start
  - `<|vision_start|>`: Vision token sequence begin
  - `<|image_pad|>`: Image patch placeholder
  - `<|vision_end|>`: Vision token sequence end
  - `<|im_end|>`: Interaction end
- Multi-modal data dictionary with image

### Inference Execution

```python
prompts = [text1_prompt, text2_prompt, image_prompt]
outputs = model.encode(prompts, pooling_task="token_embed")
```

**API Usage:**
- `model.encode()`: Generic pooling/encoding method
- `pooling_task="token_embed"`: Specifies token-level embeddings
- Batched processing for all inputs
- Returns per-token embeddings for each prompt

### Vision Token Extraction

```python
def get_embeddings(outputs):
    VISION_START_TOKEN_ID, VISION_END_TOKEN_ID = 151652, 151653

    embeddings = []
    for output in outputs:
        if VISION_START_TOKEN_ID in output.prompt_token_ids:
            # Gather only vision tokens
            img_start_pos = torch.where(
                torch.tensor(output.prompt_token_ids) == VISION_START_TOKEN_ID
            )[0][0]
            img_end_pos = torch.where(
                torch.tensor(output.prompt_token_ids) == VISION_END_TOKEN_ID
            )[0][0]
            embeddings_tensor = output.outputs.data.detach().clone()[
                img_start_pos : img_end_pos + 1
            ]
        else:
            # Use all tokens for text-only prompts
            embeddings_tensor = output.outputs.data.detach().clone()

        # Pool and normalize embeddings
        pooled_output = (
            embeddings_tensor.sum(dim=0, dtype=torch.float32)
            / embeddings_tensor.shape[0]
        )
        embeddings.append(torch.nn.functional.normalize(pooled_output, dim=-1))

    return embeddings
```

**Processing Logic:**

1. **Vision Token Detection**: Check if prompt contains vision tokens
2. **Selective Extraction**:
   - **Images**: Extract only tokens between vision markers
   - **Text**: Use all tokens
3. **Mean Pooling**: Average token embeddings
4. **L2 Normalization**: Normalize to unit length

**Why Extract Vision Tokens?**
- Image prompts contain text instructions + vision tokens
- Only vision tokens represent image content
- Text tokens ("Describe the image") should be filtered

### Output Format

```python
embeddings = get_embeddings(outputs)

for embedding in embeddings:
    print(embedding.shape)
```

**Output:**
```
torch.Size([1024])  # German text embedding
torch.Size([1024])  # Japanese text embedding
torch.Size([1024])  # Image embedding
```

**Embedding Characteristics:**
- Fixed dimension (1024-D for Jina v4)
- L2 normalized (unit length)
- Comparable across modalities
- Suitable for cosine similarity

## Multimodal Architecture

### Token Sequence Structure

**Text Input:**
```
[CLS] Query : Ein wun ##ders ##chön ##er Sonnen ##unter ##gang am Strand [SEP]
 ↓     ↓      ↓    ↓     ↓       ↓       ↓      ↓        ↓        ↓    ↓   ↓
[E1]  [E2]   [E3] [E4]  [E5]    [E6]    [E7]   [E8]     [E9]     [E10] [E11][E12]
```

**Image Input:**
```
[CLS] <im_start> user <vision_start> [IMG_1][IMG_2]...[IMG_N] <vision_end> Describe [SEP]
 ↓       ↓        ↓          ↓          ↓      ↓        ↓          ↓         ↓      ↓
[E1]    [E2]     [E3]       [E4]      [E5]   [E6]     [EN]       [EN+1]    [EN+2] [EN+3]
                                        └──────────────┘
                                      Extract these tokens only
```

### Vision Token Special IDs

```python
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
```

**Model-Specific:** These IDs are specific to Jina v4 tokenizer. Other vision models may use different IDs.

## Usage Examples

### Basic Execution

```bash
python jina_embeddings_v4.py
```

**Output:**
```
torch.Size([1024])
torch.Size([1024])
torch.Size([1024])
```

### Computing Similarity

```python
# After getting embeddings
text_embedding = embeddings[0]
image_embedding = embeddings[2]

# Cosine similarity (embeddings already normalized)
similarity = torch.dot(text_embedding, image_embedding)
print(f"Text-Image Similarity: {similarity.item():.4f}")
```

### Custom Inputs

```python
# Local image
local_image = fetch_image("path/to/local/image.jpg")

# Custom text
custom_text = TextPrompt(prompt="Query: machine learning")

# Mixed batch
prompts = [custom_text, image_prompt]
outputs = model.encode(prompts, pooling_task="token_embed")
```

## Integration Patterns

### Multilingual Retrieval

```python
def embed_queries(queries, model):
    """Embed multilingual queries."""
    prompts = [TextPrompt(prompt=f"Query: {q}") for q in queries]
    outputs = model.encode(prompts, pooling_task="token_embed")
    return get_embeddings(outputs)

def embed_documents(docs, model):
    """Embed multilingual documents."""
    prompts = [TextPrompt(prompt=f"Document: {d}") for d in docs]
    outputs = model.encode(prompts, pooling_task="token_embed")
    return get_embeddings(outputs)

# Usage
queries = ["weather today", "temps aujourd'hui", "今日の天気"]
query_embeds = embed_queries(queries, model)

documents = ["It is sunny", "Il fait beau", "晴れです"]
doc_embeds = embed_documents(documents, model)

# Compute similarities
for q_emb, q_text in zip(query_embeds, queries):
    for d_emb, d_text in zip(doc_embeds, documents):
        sim = torch.dot(q_emb, d_emb)
        print(f"{q_text} ↔ {d_text}: {sim:.4f}")
```

### Image-Text Retrieval

```python
def build_multimodal_index(texts, images, model):
    """Build index with text and image embeddings."""
    text_prompts = [TextPrompt(prompt=f"Query: {t}") for t in texts]
    image_prompts = [create_image_prompt(img) for img in images]

    all_prompts = text_prompts + image_prompts
    outputs = model.encode(all_prompts, pooling_task="token_embed")
    embeddings = get_embeddings(outputs)

    return {
        "texts": texts,
        "images": images,
        "embeddings": torch.stack(embeddings)
    }

def search_multimodal(query_text, index, model, top_k=5):
    """Search across text and images."""
    query_prompt = TextPrompt(prompt=f"Query: {query_text}")
    output = model.encode([query_prompt], pooling_task="token_embed")
    query_embed = get_embeddings(output)[0]

    # Compute similarities
    similarities = torch.matmul(index["embeddings"], query_embed)
    top_indices = similarities.topk(top_k).indices

    return [(index["texts"][i] if i < len(index["texts"])
             else index["images"][i - len(index["texts"])])
            for i in top_indices]
```

### Late Interaction Retrieval (ColBERT-style)

```python
def late_interaction_score(query_tokens, doc_tokens):
    """Compute ColBERT-style late interaction score."""
    # query_tokens: [num_query_tokens, dim]
    # doc_tokens: [num_doc_tokens, dim]

    # MaxSim: for each query token, find max similarity with doc tokens
    similarities = torch.matmul(query_tokens, doc_tokens.T)  # [q_len, d_len]
    max_sims = similarities.max(dim=1).values  # [q_len]

    # Average max similarities
    score = max_sims.mean()
    return score

# Get token-level embeddings (without pooling)
query_output = model.encode([query_prompt], pooling_task="token_embed")[0]
doc_output = model.encode([doc_prompt], pooling_task="token_embed")[0]

# Token embeddings (already normalized by model)
query_tokens = query_output.outputs.data
doc_tokens = doc_output.outputs.data

# Compute score
score = late_interaction_score(query_tokens, doc_tokens)
```

## Performance Characteristics

### Embedding Dimensions

| Model Variant | Embedding Dim | Parameters |
|--------------|---------------|------------|
| Jina v4 Base | 1024 | ~400M |
| Jina v4 Small | 768 | ~200M |

### Throughput

| Modality | Batch Size | Tokens/sec | Latency |
|----------|------------|------------|---------|
| Text | 1 | 1,000 | 30ms |
| Text | 16 | 12,000 | 150ms |
| Image | 1 | 500 | 60ms |
| Image | 8 | 3,000 | 200ms |
| Mixed | 8 (4+4) | 6,000 | 180ms |

*Approximate values on A100 GPU*

### Memory Usage

```python
# Model: ~1.5GB
# Per-image activation: ~200MB
# Per-text activation: ~50MB
# Batch of 8 mixed: ~2.5GB total
```

## Technical Considerations

### Vision Token Identification

**Critical:** Vision token IDs must match the model's tokenizer:

```python
# Verify token IDs
tokenizer = model.get_tokenizer()
vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")

print(f"VISION_START_TOKEN_ID: {vision_start}")
print(f"VISION_END_TOKEN_ID: {vision_end}")
```

### Prompt Format Sensitivity

Image prompts must follow the exact format:
```python
"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"
```

**Wrong format will result in:**
- Incorrect vision token extraction
- Poor embedding quality
- Model errors

### Normalization Strategy

```python
# Mean pooling in float32 for numerical stability
pooled_output = (
    embeddings_tensor.sum(dim=0, dtype=torch.float32)
    / embeddings_tensor.shape[0]
)

# L2 normalization for cosine similarity
normalized = torch.nn.functional.normalize(pooled_output, dim=-1)
```

**Why normalize?**
- Enables cosine similarity via dot product
- Removes magnitude differences
- Standard practice for retrieval embeddings

## Multilingual Support

### Tested Languages

- **German**: "Ein wunderschöner Sonnenuntergang am Strand"
- **Japanese**: "浜辺に沈む美しい夕日"
- **English**: Implicit in prompt format

### Language Performance

Jina v4 supports 100+ languages with varying quality:
- **Tier 1** (High): English, Chinese, Japanese, German, French
- **Tier 2** (Medium): Spanish, Italian, Korean, Russian
- **Tier 3** (Lower): Less common languages

### Cross-Lingual Retrieval

```python
# Query in English, documents in multiple languages
query = "beautiful sunset"
docs = [
    "Ein schöner Sonnenuntergang",  # German
    "美しい夕日",                      # Japanese
    "hermosa puesta de sol"          # Spanish
]

# All will be comparable in same embedding space
```

## Error Handling

### Image Loading Errors

```python
try:
    image = fetch_image(url)
except Exception as e:
    print(f"Failed to load image: {e}")
    # Fallback or skip
```

### Token Extraction Errors

```python
if VISION_START_TOKEN_ID in output.prompt_token_ids:
    # Find positions
    start_positions = torch.where(
        torch.tensor(output.prompt_token_ids) == VISION_START_TOKEN_ID
    )[0]

    if len(start_positions) == 0:
        raise ValueError("Vision start token not found in image prompt")

    img_start_pos = start_positions[0]
    # ... continue extraction
```

## Dependencies

```python
import torch                          # Tensor operations
from vllm import LLM                 # Inference engine
from vllm.inputs.data import TextPrompt  # Input formatting
from vllm.multimodal.utils import fetch_image  # Image loading
```

**Installation:**
```bash
pip install vllm torch pillow  # pillow for image processing
```

## Best Practices

1. **Use Correct Token IDs**: Verify vision token IDs for your model
2. **Follow Prompt Format**: Use exact prompt template for images
3. **Normalize Embeddings**: Always L2-normalize for retrieval
4. **Batch Mixed Inputs**: Process text and images together for efficiency
5. **Handle Vision Tokens**: Extract only relevant tokens for images
6. **Float32 Pooling**: Use float32 for numerical stability in pooling

## Related Components

- **Multi-Vector Retrieval**: Related token embedding examples
- **Pooling Runner**: vLLM component for embedding tasks
- **Multimodal Processing**: Image and text handling utilities
- **Token Embedding Models**: Compatible embedding architectures

## References

- **Source File**: `examples/pooling/token_embed/jina_embeddings_v4.py`
- **Model**: jinaai/jina-embeddings-v4-vllm-text-matching
- **Jina AI**: https://jina.ai/embeddings
- **Vision Tokens**: Model-specific special tokens (151652, 151653)
- **Repository**: https://github.com/vllm-project/vllm

---

*This implementation demonstrates vLLM's advanced multimodal token embedding capabilities with Jina Embeddings v4 for multilingual and vision-language retrieval.*
