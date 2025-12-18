{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Reranking]], [[domain::API Compatibility]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Client demonstrating Cohere SDK compatibility with vLLM's reranking API using both v1 and v2 client versions.

=== Description ===
This example shows how vLLM's /rerank endpoint is compatible with the Cohere Python SDK, allowing seamless integration with existing Cohere-based applications. It demonstrates using both Cohere's Client (v1) and ClientV2 APIs to rerank documents based on query relevance. The example uses a BGE reranker model served by vLLM, showing that you can swap Cohere's API with vLLM-hosted models without changing application code beyond the base URL.

=== Usage ===
Use this example when migrating from Cohere's reranking API to self-hosted vLLM, when building retrieval systems that need document reranking, or when you want to use Cohere's SDK with custom reranking models. It's particularly valuable for RAG applications that need to reorder retrieved documents by relevance.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/cohere_rerank_client.py examples/pooling/score/cohere_rerank_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Install Cohere SDK
pip install cohere

# Start vLLM server with reranking model
vllm serve BAAI/bge-reranker-base

# Run Cohere-compatible client
python cohere_rerank_client.py
</syntaxhighlight>

== Key Concepts ==

=== Cohere API Compatibility ===
vLLM's /rerank endpoint implements Cohere's reranking API specification, enabling drop-in replacement with the Cohere SDK by changing only the base_url parameter.

=== Client Version Support ===
The example demonstrates both Cohere Client (v1) and ClientV2, showing that vLLM supports multiple versions of the Cohere SDK.

=== Reranking Task ===
Given a query and list of documents, the reranker scores each document's relevance to the query and returns them in ranked order with relevance scores.

=== Model Flexibility ===
While using Cohere's SDK interface, you can use any reranking model supported by vLLM, not just Cohere's proprietary models.

== Usage Examples ==

<syntaxhighlight lang="python">
import cohere
from cohere import Client, ClientV2

model = "BAAI/bge-reranker-base"
query = "What is the capital of France?"
documents = [
    "The capital of France is Paris",
    "Reranking is fun!",
    "vLLM is an open-source framework for fast AI serving"
]

# Using Cohere v1 client
cohere_v1 = cohere.Client(
    base_url="http://localhost:8000",
    api_key="sk-fake-key"  # Any key works with vLLM
)
rerank_v1_result = cohere_v1.rerank(
    model=model,
    query=query,
    documents=documents
)
print("Cohere v1 result:", rerank_v1_result)

# Using Cohere v2 client
cohere_v2 = cohere.ClientV2(
    "sk-fake-key",
    base_url="http://localhost:8000"
)
rerank_v2_result = cohere_v2.rerank(
    model=model,
    query=query,
    documents=documents
)
print("Cohere v2 result:", rerank_v2_result)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[implements::Component:vLLM_Rerank_API]]
* [[uses::Tool:Cohere_SDK]]
* [[related::Implementation:vllm-project_vllm_OpenAI_Reranker]]
* [[related::Implementation:vllm-project_vllm_Cross_Encoder_Scoring]]
