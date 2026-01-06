{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Embeddings]], [[domain::Multi-Vector Retrieval]], [[domain::API]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
HTTP client demonstrating multi-vector embeddings through vLLM's pooling API for token-level representations.

=== Description ===
This minimal client example shows how to request multi-vector embeddings via vLLM's /pooling endpoint. Unlike standard embedding APIs that return a single vector per input, this endpoint returns a matrix of token-level embeddings when the model and configuration support multi-vector retrieval. The client sends a batch of texts and receives tensor data that can be reshaped into per-token embedding matrices. This enables HTTP-based access to advanced retrieval techniques like late interaction and MaxSim scoring.

=== Usage ===
Use this example when building retrieval services with HTTP APIs that need token-level embeddings, when implementing ColBERT-style late interaction systems via REST APIs, or when you need multi-vector embeddings in web services or microservices architectures.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/token_embed/multi_vector_retrieval_client.py examples/pooling/token_embed/multi_vector_retrieval_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server with multi-vector capable model
vllm serve BAAI/bge-m3

# Run multi-vector API client
python multi_vector_retrieval_client.py

# With custom configuration
python multi_vector_retrieval_client.py \
  --host localhost \
  --port 8000 \
  --model BAAI/bge-m3
</syntaxhighlight>

== Key Concepts ==

=== Pooling API Endpoint ===
The /pooling endpoint can return different types of outputs based on the model and task, including token-level embedding matrices.

=== Batch Processing ===
Multiple texts can be processed in a single API call, with each receiving its own multi-vector embedding matrix.

=== Tensor Response Format ===
The API returns embedding data as nested lists that can be converted to PyTorch or NumPy tensors for downstream processing.

=== Shape Information ===
Each response includes shape information showing [num_tokens, embedding_dim], indicating how many token embeddings were generated.

=== HTTP Overhead ===
While convenient, HTTP-based multi-vector embeddings involve more data transfer than single embeddings, so consider batching and network optimization.

== Usage Examples ==

<syntaxhighlight lang="python">
import argparse
import requests
import torch

# Configuration
api_url = "http://localhost:8000/pooling"
model_name = "BAAI/bge-m3"

# Prepare batch of texts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is"
]

# Create request payload
prompt = {
    "model": model_name,
    "input": prompts
}

# Send request
headers = {"User-Agent": "Test Client"}
response = requests.post(api_url, headers=headers, json=prompt)

# Process multi-vector embeddings
for output in response.json()["data"]:
    # Convert to tensor
    multi_vector = torch.tensor(output["data"])

    # Display shape (tokens × embedding_dim)
    print(f"Multi-vector shape: {multi_vector.shape}")

    # Each row is a token embedding
    # Can be used for late interaction retrieval
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[related::Implementation:vllm-project_vllm_Multi_Vector_Embeddings]]
* [[related::Implementation:vllm-project_vllm_Dual_Format_Pooling_Client]]
