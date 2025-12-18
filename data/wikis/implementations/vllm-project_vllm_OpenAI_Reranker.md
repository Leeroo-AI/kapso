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
Minimal HTTP client demonstrating raw reranking API compatible with both Jina and Cohere rerank endpoints.

=== Description ===
This lightweight example shows direct HTTP usage of vLLM's /rerank endpoint without SDK dependencies. It sends a query and list of documents to the endpoint and receives relevance-ranked results. The API is compatible with both Jina AI's and Cohere's reranking API specifications, allowing drop-in replacement for either service. This raw HTTP approach is useful for understanding the underlying API structure or when you want to avoid additional dependencies.

=== Usage ===
Use this example when you need simple reranking without external SDKs, when integrating with systems that use raw HTTP, or when learning the rerank API structure. It's ideal for lightweight applications, serverless functions, or when you want minimal dependencies.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/openai_reranker.py examples/pooling/score/openai_reranker.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server with reranking model
vllm serve BAAI/bge-reranker-base

# Run raw HTTP reranker client
python openai_reranker.py
</syntaxhighlight>

== Key Concepts ==

=== Raw HTTP Interface ===
Uses Python's requests library directly without wrapper SDKs, providing clear visibility into the actual HTTP request structure.

=== Jina/Cohere Compatibility ===
The /rerank endpoint implements the API specification used by both Jina AI and Cohere, enabling compatibility with tools expecting either format.

=== Request Structure ===
The request payload contains: model (model identifier), query (search query), and documents (list of candidate documents to rank).

=== Response Format ===
The API returns documents sorted by relevance score, typically with each result including the document, score, and original index.

== Usage Examples ==

<syntaxhighlight lang="python">
import json
import requests

# Configuration
url = "http://127.0.0.1:8000/rerank"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

# Prepare reranking request
data = {
    "model": "BAAI/bge-reranker-base",
    "query": "What is the capital of France?",
    "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Horses and cows are both animals"
    ]
}

# Send request
response = requests.post(url, headers=headers, json=data)

# Check response
if response.status_code == 200:
    print("Request successful!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[implements::Component:vLLM_Rerank_API]]
* [[related::Implementation:vllm-project_vllm_Cohere_Rerank_Client]]
* [[related::Implementation:vllm-project_vllm_Cross_Encoder_Scoring]]
