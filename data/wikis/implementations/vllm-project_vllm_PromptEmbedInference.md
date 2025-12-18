{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Online Serving]], [[domain::Embeddings]], [[domain::Advanced Features]], [[domain::OpenAI API]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates how to send pre-computed prompt embeddings to vLLM's OpenAI-compatible API, enabling advanced use cases like cached embeddings or custom embedding models.

=== Description ===
This example shows how to use vLLM's prompt embeddings feature, which allows sending pre-computed embeddings instead of text prompts. The workflow involves:
1. Computing prompt embeddings using HuggingFace Transformers
2. Encoding embeddings to base64 format
3. Sending encoded embeddings via OpenAI-compatible API's <code>extra_body</code> parameter

This advanced pattern enables several sophisticated use cases:
* Caching embeddings for frequently-used prompt prefixes
* Using different embedding models than the generation model
* Implementing custom prompt processing or manipulation in embedding space
* Bypassing tokenization for certain workflows
* Building hybrid retrieval-generation systems

The example demonstrates this with Llama 3.2 1B Instruct, but the pattern works with any model supporting prompt embeddings.

=== Usage ===
Use this approach when:
* Building systems with cached common prompt prefixes
* Implementing prompt compression or manipulation techniques
* Using custom or external embedding models
* Developing hybrid RAG systems with specialized retrievers
* Optimizing latency by pre-computing embeddings
* Implementing prompt injection defenses through embedding validation
* Building experimental architectures that manipulate embedding spaces

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/prompt_embed_inference_with_openai_client.py examples/online_serving/prompt_embed_inference_with_openai_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server with prompt embeddings enabled
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --runner generate \
    --max-model-len 4096 \
    --enable-prompt-embeds

# In another terminal, run the example
python examples/online_serving/prompt_embed_inference_with_openai_client.py

# Output will show completion generated from embeddings
</syntaxhighlight>

== Key Concepts ==

=== Prompt Embeddings ===
Instead of sending text:
<syntaxhighlight lang="python">
completion = client.completions.create(
    prompt="What is the capital of France?",
    ...
)
</syntaxhighlight>

Send pre-computed embeddings:
<syntaxhighlight lang="python">
completion = client.completions.create(
    prompt="",  # Empty string required by OpenAI client
    extra_body={"prompt_embeds": encoded_embeddings},
    ...
)
</syntaxhighlight>

The embeddings bypass tokenization and go directly to the model's transformer layers.

=== Computing Embeddings ===
Use HuggingFace Transformers to compute embeddings:
<syntaxhighlight lang="python">
import transformers

# Load tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize prompt
token_ids = tokenizer.apply_chat_template(
    chat, add_generation_prompt=True, return_tensors="pt"
)

# Get embeddings from embedding layer
embedding_layer = model.get_input_embeddings()
prompt_embeds = embedding_layer(token_ids).squeeze(0)
</syntaxhighlight>

=== Encoding Format ===
Embeddings must be base64-encoded before sending:
<syntaxhighlight lang="python">
from vllm.utils.serial_utils import tensor2base64

# Convert tensor to base64 string
encoded_embeds = tensor2base64(prompt_embeds)

# Send via extra_body
completion = client.completions.create(
    model=model_name,
    prompt="",
    extra_body={"prompt_embeds": encoded_embeds},
    ...
)
</syntaxhighlight>

=== Server Requirements ===
The server must be started with <code>--enable-prompt-embeds</code> flag:
<syntaxhighlight lang="bash">
vllm serve model_name --enable-prompt-embeds
</syntaxhighlight>

Without this flag, the server will reject requests with prompt embeddings.

== Usage Examples ==

=== Basic Prompt Embedding Inference ===
<syntaxhighlight lang="python">
import transformers
from openai import OpenAI
from vllm.utils.serial_utils import tensor2base64

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load transformers model to compute embeddings
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
transformers_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

# Prepare chat prompt
chat = [{"role": "user", "content": "Please tell me about the capital of France."}]

# Tokenize with chat template
token_ids = tokenizer.apply_chat_template(
    chat, add_generation_prompt=True, return_tensors="pt"
)

# Compute embeddings
embedding_layer = transformers_model.get_input_embeddings()
prompt_embeds = embedding_layer(token_ids).squeeze(0)

# Encode to base64
encoded_embeds = tensor2base64(prompt_embeds)

# Generate with embeddings
completion = client.completions.create(
    model=model_name,
    prompt="",  # Required but empty
    max_tokens=50,
    temperature=0.0,
    extra_body={"prompt_embeds": encoded_embeds},
)

print(completion.choices[0].text)
</syntaxhighlight>

=== Caching Common Prefixes ===
<syntaxhighlight lang="python">
import torch
from vllm.utils.serial_utils import tensor2base64

class EmbeddingCache:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.embedding_layer = model.get_input_embeddings()
        self.cache = {}

    def get_system_embedding(self, system_prompt):
        """Cache system prompt embeddings"""
        if system_prompt not in self.cache:
            messages = [{"role": "system", "content": system_prompt}]
            token_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, return_tensors="pt"
            )
            embeds = self.embedding_layer(token_ids).squeeze(0)
            self.cache[system_prompt] = embeds
        return self.cache[system_prompt]

    def combine_with_user_prompt(self, system_prompt, user_prompt):
        """Combine cached system embedding with new user prompt"""
        # Get cached system embedding
        system_embeds = self.get_system_embedding(system_prompt)

        # Compute user prompt embedding
        messages = [{"role": "user", "content": user_prompt}]
        user_tokens = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        user_embeds = self.embedding_layer(user_tokens).squeeze(0)

        # Concatenate embeddings
        combined = torch.cat([system_embeds, user_embeds], dim=0)
        return tensor2base64(combined)

# Usage
cache = EmbeddingCache(tokenizer, transformers_model)

system = "You are a helpful assistant expert in geography."

# Process multiple user prompts with cached system prompt
for user_prompt in ["Capital of France?", "Capital of Spain?", "Capital of Italy?"]:
    encoded = cache.combine_with_user_prompt(system, user_prompt)

    completion = client.completions.create(
        model=model_name,
        prompt="",
        max_tokens=50,
        extra_body={"prompt_embeds": encoded},
    )
    print(f"Q: {user_prompt}")
    print(f"A: {completion.choices[0].text}\n")
</syntaxhighlight>

=== Custom Embedding Manipulation ===
<syntaxhighlight lang="python">
import torch

def apply_prompt_style_transfer(prompt_embeds, style_vector, alpha=0.1):
    """Apply style transfer in embedding space"""
    # Add style vector to embeddings
    styled_embeds = prompt_embeds + alpha * style_vector
    return styled_embeds

# Compute embeddings for two different styles
formal_prompt = [{"role": "user", "content": "Please explain quantum computing."}]
casual_prompt = [{"role": "user", "content": "Hey, tell me about quantum computing!"}]

formal_tokens = tokenizer.apply_chat_template(formal_prompt, return_tensors="pt")
casual_tokens = tokenizer.apply_chat_template(casual_prompt, return_tensors="pt")

formal_embeds = embedding_layer(formal_tokens).squeeze(0)
casual_embeds = embedding_layer(casual_tokens).squeeze(0)

# Compute style vector (difference between formal and casual)
style_vector = formal_embeds - casual_embeds

# Apply style transfer to new prompt
base_prompt = [{"role": "user", "content": "Describe machine learning."}]
base_tokens = tokenizer.apply_chat_template(base_prompt, return_tensors="pt")
base_embeds = embedding_layer(base_tokens).squeeze(0)

# Generate with different styles
for alpha in [0.0, 0.5, 1.0]:
    styled = apply_prompt_style_transfer(base_embeds, style_vector, alpha)
    encoded = tensor2base64(styled)

    completion = client.completions.create(
        model=model_name,
        prompt="",
        max_tokens=100,
        extra_body={"prompt_embeds": encoded},
    )
    print(f"Style alpha={alpha}: {completion.choices[0].text}\n")
</syntaxhighlight>

=== Embedding Interpolation ===
<syntaxhighlight lang="python">
def interpolate_prompts(prompt1, prompt2, alpha=0.5):
    """Interpolate between two prompts in embedding space"""
    # Compute embeddings for both prompts
    tokens1 = tokenizer.apply_chat_template(prompt1, return_tensors="pt")
    tokens2 = tokenizer.apply_chat_template(prompt2, return_tensors="pt")

    embeds1 = embedding_layer(tokens1).squeeze(0)
    embeds2 = embedding_layer(tokens2).squeeze(0)

    # Ensure same length (pad shorter one)
    max_len = max(embeds1.size(0), embeds2.size(0))
    if embeds1.size(0) < max_len:
        padding = torch.zeros(max_len - embeds1.size(0), embeds1.size(1))
        embeds1 = torch.cat([embeds1, padding], dim=0)
    if embeds2.size(0) < max_len:
        padding = torch.zeros(max_len - embeds2.size(0), embeds2.size(1))
        embeds2 = torch.cat([embeds2, padding], dim=0)

    # Interpolate
    interpolated = alpha * embeds1 + (1 - alpha) * embeds2
    return tensor2base64(interpolated)

# Interpolate between technical and creative prompts
technical = [{"role": "user", "content": "Explain neural networks formally."}]
creative = [{"role": "user", "content": "Describe neural networks like a story."}]

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    encoded = interpolate_prompts(technical, creative, alpha)

    completion = client.completions.create(
        model=model_name,
        prompt="",
        max_tokens=100,
        extra_body={"prompt_embeds": encoded},
    )
    print(f"Alpha={alpha}: {completion.choices[0].text}\n")
</syntaxhighlight>

=== Using External Embedding Model ===
<syntaxhighlight lang="python">
# Use a different model for embeddings than for generation
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
generation_model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load embedding model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(embedding_model_name)

# Load generation model's embedding layer to match dimensions
gen_tokenizer = transformers.AutoTokenizer.from_pretrained(generation_model_name)
gen_model = transformers.AutoModelForCausalLM.from_pretrained(generation_model_name)
gen_embedding_layer = gen_model.get_input_embeddings()

# Project external embeddings to generation model's space
import torch.nn as nn

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

# Train or use pre-trained projector (simplified here)
projector = EmbeddingProjector(384, gen_embedding_layer.embedding_dim)

# Compute and project embeddings
external_embeds = embedding_model.encode("What is machine learning?", convert_to_tensor=True)
projected_embeds = projector(external_embeds.unsqueeze(0))
encoded = tensor2base64(projected_embeds.squeeze(0))

# Generate
completion = client.completions.create(
    model=generation_model_name,
    prompt="",
    max_tokens=100,
    extra_body={"prompt_embeds": encoded},
)
</syntaxhighlight>

=== Batch Processing with Cached Embeddings ===
<syntaxhighlight lang="python">
import pickle
from pathlib import Path

class PersistentEmbeddingCache:
    def __init__(self, cache_file="embedding_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache = self.load_cache()

    def load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f)

    def get_or_compute(self, prompt, tokenizer, embedding_layer):
        if prompt not in self.cache:
            messages = [{"role": "user", "content": prompt}]
            tokens = tokenizer.apply_chat_template(messages, return_tensors="pt")
            embeds = embedding_layer(tokens).squeeze(0)
            self.cache[prompt] = tensor2base64(embeds)
            self.save_cache()
        return self.cache[prompt]

# Use persistent cache
cache = PersistentEmbeddingCache()

prompts = [
    "What is AI?",
    "Explain machine learning.",
    "What is deep learning?",
]

for prompt in prompts:
    encoded = cache.get_or_compute(prompt, tokenizer, embedding_layer)

    completion = client.completions.create(
        model=model_name,
        prompt="",
        max_tokens=50,
        extra_body={"prompt_embeds": encoded},
    )
    print(f"Q: {prompt}")
    print(f"A: {completion.choices[0].text}\n")
</syntaxhighlight>

== Advanced Use Cases ==

=== Prompt Compression ===
Reduce prompt length by averaging embeddings:
<syntaxhighlight lang="python">
def compress_prompt_embeddings(prompt_embeds, target_length):
    """Compress embeddings to target length"""
    current_length = prompt_embeds.size(0)
    if current_length <= target_length:
        return prompt_embeds

    # Simple average pooling
    pool_size = current_length // target_length
    compressed = torch.nn.functional.avg_pool1d(
        prompt_embeds.unsqueeze(0).transpose(1, 2),
        kernel_size=pool_size,
        stride=pool_size
    ).transpose(1, 2).squeeze(0)

    return compressed[:target_length]
</syntaxhighlight>

=== Security: Embedding Validation ===
Validate embeddings before generation to detect prompt injection:
<syntaxhighlight lang="python">
def is_safe_embedding(embedding, reference_embeddings, threshold=0.9):
    """Check if embedding is similar to known safe embeddings"""
    from torch.nn.functional import cosine_similarity

    for ref in reference_embeddings:
        sim = cosine_similarity(embedding, ref, dim=-1).mean()
        if sim > threshold:
            return True
    return False
</syntaxhighlight>

== Performance Considerations ==

=== Latency ===
* Computing embeddings adds latency (tokenization + embedding lookup)
* Caching embeddings eliminates this overhead for repeated prompts
* Network transfer of base64 embeddings is slightly larger than text
* Overall latency similar or better with caching

=== Memory ===
* Embeddings: <code>sequence_length × hidden_dim × 4 bytes</code>
* Example: 100 tokens × 4096 dim × 4 bytes = ~1.6MB per prompt
* Base64 encoding increases size by ~33%
* Caching many embeddings requires significant memory

== Limitations ==

=== Server Configuration ===
* Requires <code>--enable-prompt-embeds</code> flag
* Not all models support prompt embeddings
* May have compatibility issues with certain features

=== API Compatibility ===
* Requires <code>extra_body</code> parameter (OpenAI client extension)
* Not part of standard OpenAI API
* May not work with all OpenAI-compatible clients

=== Embedding Dimensions ===
* Embeddings must match model's expected dimensions
* Cannot use embeddings from incompatible models without projection
* Sequence length limits still apply

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
