{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::NER]], [[domain::Token Classification]], [[domain::API]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
HTTP client for Named Entity Recognition using vLLM's pooling API with token-level classification output.

=== Description ===
This example demonstrates online NER inference through vLLM's /pooling endpoint with a token classification model. It shows the complete workflow: sending text to the API, receiving per-token logits, using Transformers to map token IDs to tokens, and displaying entity predictions. The client-side tokenization ensures alignment between the API's token predictions and the displayed tokens, making it suitable for real-time NER services.

=== Usage ===
Use this example when building NER services with HTTP APIs, when you need real-time entity extraction, or when integrating NER into web applications. It's ideal for microservices architectures where NER runs as a separate service accessed via HTTP.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/token_classify/ner_client.py examples/pooling/token_classify/ner_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Start vLLM server with NER model
vllm serve boltuix/NeuroBERT-NER

# Run NER API client
python ner_client.py

# With custom configuration
python ner_client.py \
  --host localhost \
  --port 8000 \
  --model boltuix/NeuroBERT-NER
</syntaxhighlight>

== Key Concepts ==

=== Pooling API for Token Classification ===
The /pooling endpoint returns per-token logits when used with token classification models, enabling token-level predictions through a standard HTTP interface.

=== Client-Side Tokenization ===
The client tokenizes the input text using Transformers to ensure exact alignment between token predictions from the API and the displayed tokens.

=== Label ID Mapping ===
Uses the model's AutoConfig to load the id2label mapping, converting numeric predictions to entity tags like B-PER (beginning of person), I-ORG (inside organization), etc.

=== Tensor Conversion ===
API responses contain lists of logits which are converted to PyTorch tensors for argmax operations to get predicted labels.

=== BIO Tagging ===
NER models typically use BIO (Beginning-Inside-Outside) or IOB tagging schemes, where B- marks entity start, I- marks continuation, and O marks non-entities.

== Usage Examples ==

<syntaxhighlight lang="python">
import argparse
import requests
import torch
from transformers import AutoConfig, AutoTokenizer

# Configuration
api_url = "http://localhost:8000/pooling"
model_name = "boltuix/NeuroBERT-NER"

# Load tokenizer and config
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
label_map = config.id2label

# Prepare input
text = "Barack Obama visited Microsoft headquarters in Seattle on January 2025."
prompt = {
    "model": model_name,
    "input": text
}

# Send request
headers = {"User-Agent": "Test Client"}
response = requests.post(api_url, headers=headers, json=prompt)

# Process response
output = response.json()["data"][0]
logits = torch.tensor(output["data"])
predictions = logits.argmax(dim=-1)

# Tokenize for display
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
labels = [label_map[p.item()] for p in predictions]

assert len(tokens) == len(predictions), "Token-prediction mismatch"

# Display results
for token, label in zip(tokens, labels):
    if token not in tokenizer.all_special_tokens:
        print(f"{token:15} → {label}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[implements::Component:vLLM_Pooling_API]]
* [[uses::Tool:Transformers]]
* [[related::Implementation:vllm-project_vllm_Offline_NER_Example]]
* [[related::Implementation:vllm-project_vllm_Classification_API_Client]]
