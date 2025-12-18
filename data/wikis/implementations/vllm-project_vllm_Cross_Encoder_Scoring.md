{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Cross-Encoder]], [[domain::Scoring]], [[domain::Reranking]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
HTTP client demonstrating cross-encoder similarity scoring with flexible input formats for query-document pairs.

=== Description ===
This example showcases vLLM's /score endpoint for cross-encoder models that compute semantic similarity scores between text pairs. It demonstrates three input patterns: (1) single query vs single document, (2) single query vs multiple documents, and (3) paired lists of queries and documents. Cross-encoders jointly encode both texts to produce accurate similarity scores, making them ideal for reranking and semantic matching tasks. The example uses BGE reranker models which are optimized for relevance scoring.

=== Usage ===
Use this example when implementing semantic search reranking, document relevance scoring, or any task requiring precise similarity measurements between text pairs. It's particularly useful for second-stage ranking in retrieval systems where you need high accuracy on a smaller candidate set.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/openai_cross_encoder_score.py examples/pooling/score/openai_cross_encoder_score.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server with cross-encoder model
vllm serve BAAI/bge-reranker-v2-m3

# Run scoring client demonstrating all input patterns
python openai_cross_encoder_score.py

# With custom configuration
python openai_cross_encoder_score.py \
  --host localhost \
  --port 8000 \
  --model BAAI/bge-reranker-v2-m3
</syntaxhighlight>

== Key Concepts ==

=== Cross-Encoder Architecture ===
Unlike bi-encoders that encode texts independently, cross-encoders process query and document together, allowing attention mechanisms to model their interaction for more accurate scoring.

=== Flexible Input Formats ===
The /score endpoint accepts three patterns: (1) text_1: str, text_2: str for single pairs, (2) text_1: str, text_2: List[str] for one-to-many scoring, (3) text_1: List[str], text_2: List[str] for batch pairwise scoring.

=== One-to-Many Scoring ===
When text_2 is a list, the API scores a single query against multiple documents in one request, enabling efficient batch reranking.

=== Pairwise Batch Scoring ===
When both text_1 and text_2 are lists of equal length, the API computes scores for corresponding pairs (text_1[i], text_2[i]), useful for evaluating multiple query-document combinations.

== Usage Examples ==

<syntaxhighlight lang="python">
import requests
import pprint

api_url = "http://localhost:8000/score"
model_name = "BAAI/bge-reranker-v2-m3"
headers = {"User-Agent": "Test Client"}

# Example 1: Single query vs single document
text_1 = "What is the capital of Brazil?"
text_2 = "The capital of Brazil is Brasilia."
prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}

response = requests.post(api_url, headers=headers, json=prompt)
print("\nPrompt when text_1 and text_2 are both strings:")
pprint.pprint(prompt)
print("\nScore Response:")
pprint.pprint(response.json())

# Example 2: Single query vs multiple documents (reranking)
text_1 = "What is the capital of France?"
text_2 = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris."
]
prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}

response = requests.post(api_url, headers=headers, json=prompt)
print("\nPrompt when text_1 is string and text_2 is a list:")
pprint.pprint(prompt)
print("\nScore Response:")
pprint.pprint(response.json())

# Example 3: Batch pairwise scoring
text_1 = [
    "What is the capital of Brazil?",
    "What is the capital of France?"
]
text_2 = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris."
]
prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}

response = requests.post(api_url, headers=headers, json=prompt)
print("\nPrompt when text_1 and text_2 are both lists:")
pprint.pprint(prompt)
print("\nScore Response:")
pprint.pprint(response.json())
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[implements::Component:vLLM_Score_API]]
* [[related::Implementation:vllm-project_vllm_Cohere_Rerank_Client]]
* [[related::Implementation:vllm-project_vllm_OpenAI_Reranker]]
* [[related::Implementation:vllm-project_vllm_CausalLM_to_SeqCls_Converter]]
