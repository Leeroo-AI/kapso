{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Embeddings]], [[domain::Multi-Vector Retrieval]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Offline example demonstrating multi-vector embeddings where each input produces multiple embedding vectors per token.

=== Description ===
This example shows the difference between standard embeddings and multi-vector (token-level) embeddings using vLLM. It first uses llm.embed() to generate single embeddings per input, then uses llm.encode() with pooling_task="token_embed" to generate per-token embeddings. Multi-vector retrieval can provide finer-grained semantic matching by preserving token-level information rather than pooling everything into a single vector. This technique is particularly effective for long documents or when you need to identify which specific parts of a text are relevant.

=== Usage ===
Use this example when implementing advanced retrieval systems that benefit from token-level granularity, when working with long documents where single vectors lose information, or when you need fine-grained semantic matching. It's particularly relevant for MaxSim retrieval and late interaction models like ColBERT.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/token_embed/multi_vector_retrieval.py examples/pooling/token_embed/multi_vector_retrieval.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run multi-vector embedding example
python multi_vector_retrieval.py

# With custom model
python multi_vector_retrieval.py --model BAAI/bge-m3
</syntaxhighlight>

== Key Concepts ==

=== Single vs Multi-Vector ===
Standard embeddings (llm.embed()) produce one vector per input. Multi-vector embeddings (llm.encode() with token_embed) produce a matrix of vectors, one per token.

=== Token-Level Representations ===
Each token gets its own contextualized embedding, preserving fine-grained semantic information that pooling operations would lose.

=== Late Interaction ===
Multi-vector embeddings enable late interaction retrieval where query and document token embeddings are compared directly, allowing more nuanced matching.

=== Shape Differences ===
Single embeddings have shape [embedding_dim], while multi-vector embeddings have shape [num_tokens, embedding_dim].

=== Use Cases ===
Multi-vector embeddings excel at: (1) matching specific phrases within documents, (2) handling long texts without information loss, (3) enabling MaxSim scoring where the maximum similarity across token pairs is used.

== Usage Examples ==

<syntaxhighlight lang="python">
from vllm import LLM

# Initialize model
llm = LLM(
    model="BAAI/bge-m3",
    runner="pooling",
    enforce_eager=True
)

# Sample prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is"
]

# Generate standard single embeddings
outputs = llm.embed(prompts)

print("\nStandard Embeddings (single vector per input):")
print("-" * 60)
for prompt, output in zip(prompts, outputs):
    embedding = output.outputs.embedding
    print(f"Embedding dimension: {len(embedding)}")

# Generate multi-vector embeddings (one per token)
outputs = llm.encode(prompts, pooling_task="token_embed")

print("\nMulti-Vector Embeddings (one vector per token):")
print("-" * 60)
for prompt, output in zip(prompts, outputs):
    multi_vector = output.outputs.data
    print(f"Shape: {multi_vector.shape} (tokens × embedding_dim)")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[related::Implementation:vllm-project_vllm_Multi_Vector_API_Client]]
