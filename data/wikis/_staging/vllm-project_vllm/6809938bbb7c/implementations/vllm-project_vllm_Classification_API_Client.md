{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Classification]], [[domain::Pooling API]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
HTTP client demonstrating text classification using vLLM's /classify endpoint with sequence classification models.

=== Description ===
This simple example shows how to use vLLM's classification API to perform text classification tasks. It sends a batch of text prompts to the /classify endpoint and receives classification predictions. The example uses a Korean sentiment analysis model (jason9693/Qwen2.5-1.5B-apeach) but the pattern works with any sequence classification model supported by vLLM's pooling runner.

=== Usage ===
Use this example when performing text classification tasks like sentiment analysis, topic categorization, or intent detection using models served by vLLM. It's suitable for batch classification of multiple texts and demonstrates the straightforward HTTP API for classification inference.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/classify/openai_classification_client.py examples/pooling/classify/openai_classification_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server with classification model
vllm serve jason9693/Qwen2.5-1.5B-apeach

# Run classification client
python openai_classification_client.py

# With custom server configuration
python openai_classification_client.py \
  --host localhost \
  --port 8000 \
  --model jason9693/Qwen2.5-1.5B-apeach
</syntaxhighlight>

== Key Concepts ==

=== Classification Endpoint ===
The /classify endpoint accepts a model name and list of input texts, returning classification results for each input. This is distinct from the generation or pooling endpoints.

=== Batch Processing ===
Multiple texts can be classified in a single API call by passing a list to the "input" field, enabling efficient batch inference.

=== Model Requirements ===
The model must be a sequence classification model compatible with vLLM's pooling runner. These models typically have a classification head trained for specific tasks.

=== Response Format ===
The API returns classification results including predicted classes, confidence scores, or logits depending on the model's output configuration.

== Usage Examples ==

<syntaxhighlight lang="python">
import requests
import pprint

# Configuration
api_url = "http://localhost:8000/classify"
model_name = "jason9693/Qwen2.5-1.5B-apeach"

# Prepare batch of texts to classify
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is"
]

# Create request payload
payload = {
    "model": model_name,
    "input": prompts
}

# Send classification request
headers = {"User-Agent": "Test Client"}
response = requests.post(api_url, headers=headers, json=payload)

# Display results
pprint.pprint(response.json())
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[implements::Component:vLLM_Pooling_API]]
* [[related::Implementation:vllm-project_vllm_NER_API_Client]]
* [[related::Implementation:vllm-project_vllm_Cross_Encoder_Scoring]]
