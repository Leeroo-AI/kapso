# Implementation: Named Entity Recognition Client

**File:** `/tmp/praxium_repo_583nq7ea/examples/pooling/token_classify/ner_client.py`
**Type:** HTTP Client Example
**Lines of Code:** 71
**Last Updated:** 2025-12-17

## Overview

The `ner_client.py` script demonstrates online named entity recognition using vLLM's `/pooling` endpoint with token classification models. This client shows how to consume vLLM's token classification API for production NER services with remote model inference.

### Purpose

Illustrates client-side integration with vLLM's token classification API, enabling scalable NER services through HTTP requests without local model deployment.

### Key Features

- **HTTP-Based Inference**: Remote model access via REST API
- **Client-Side Processing**: Token alignment and label mapping
- **Flexible Configuration**: Parameterized host, port, and model selection
- **Production-Ready Pattern**: Server-client architecture for NER services

## Architecture

### Client-Server Communication Flow

```
┌──────────────────┐                    ┌──────────────────┐
│   NER Client     │                    │   vLLM Server    │
│   (This File)    │                    │   (Pooling)      │
└────────┬─────────┘                    └────────┬─────────┘
         │                                       │
         │  1. POST /pooling                    │
         │     {"input": "text..."}             │
         ├──────────────────────────────────────>│
         │                                       │
         │                              2. Model inference
         │                              3. Extract logits
         │                                       │
         │  4. Return logits                    │
         │     {"data": [...]}                  │
         │<──────────────────────────────────────┤
         │                                       │
         │  5. Process locally:                 │
         │     - Tokenize input                 │
         │     - Argmax logits                  │
         │     - Map to labels                  │
         └─────────────────────────────────────────
```

### Data Flow

```
Input Text
    ↓
HTTP Request (raw text)
    ↓
[Server] Tokenization + Inference
    ↓
HTTP Response (logits tensor)
    ↓
[Client] Re-tokenization
    ↓
[Client] Argmax + Label Mapping
    ↓
Entity-Token Pairs
```

## Implementation Details

### Complete Source Code Structure

```python
import argparse
import requests
import torch
from transformers import AutoConfig, AutoTokenizer

def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    """Send HTTP request to vLLM pooling endpoint."""
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="boltuix/NeuroBERT-NER")
    return parser.parse_args()

def main(args):
    """Main execution flow."""
    # Setup
    api_url = f"http://{args.host}:{args.port}/pooling"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    label_map = config.id2label

    # Inference
    text = "Barack Obama visited Microsoft headquarters in Seattle on January 2025."
    prompt = {"model": args.model, "input": text}
    pooling_response = post_http_request(prompt=prompt, api_url=api_url)

    # Process response
    output = pooling_response.json()["data"][0]
    logits = torch.tensor(output["data"])
    predictions = logits.argmax(dim=-1)
    inputs = tokenizer(text, return_tensors="pt")

    # Map to labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_map[p.item()] for p in predictions]
    assert len(tokens) == len(predictions)

    # Display results
    for token, label in zip(tokens, labels):
        if token not in tokenizer.all_special_tokens:
            print(f"{token:15} → {label}")
```

### Key Components

#### 1. HTTP Request Handler

```python
def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response
```

**Characteristics:**
- Simple wrapper around requests library
- Custom User-Agent for identification
- JSON payload with model and input text
- Synchronous request-response

#### 2. Model Configuration Loading

```python
tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model)
label_map = config.id2label
```

**Client-Side Requirements:**
- Tokenizer for text processing
- Config for label mapping
- Must match server's model configuration

**Why Client-Side?** The server returns raw logits without label interpretation, so the client needs the label map to decode predictions.

#### 3. API Request Construction

```python
text = "Barack Obama visited Microsoft headquarters in Seattle on January 2025."
prompt = {"model": args.model, "input": text}
pooling_response = post_http_request(prompt=prompt, api_url=api_url)
```

**Request Format:**
```json
{
  "model": "boltuix/NeuroBERT-NER",
  "input": "Barack Obama visited..."
}
```

#### 4. Response Processing

```python
# Extract logits from response
output = pooling_response.json()["data"][0]
logits = torch.tensor(output["data"])
predictions = logits.argmax(dim=-1)
```

**Response Structure:**
```json
{
  "data": [
    {
      "index": 0,
      "data": [[logit_1], [logit_2], ...],  // Per-token logits
      "embedding": null
    }
  ]
}
```

**Processing Steps:**
1. Extract logits array from JSON response
2. Convert to PyTorch tensor
3. Apply argmax to get predicted label indices
4. Shape: `[seq_len, num_labels]` → `[seq_len]`

#### 5. Token Alignment

```python
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
labels = [label_map[p.item()] for p in predictions]

assert len(tokens) == len(predictions)
```

**Critical Alignment:**
- Client re-tokenizes input text
- Must use same tokenizer as server
- Token count must match logits count
- Assertion verifies alignment

#### 6. Output Generation

```python
for token, label in zip(tokens, labels):
    if token not in tokenizer.all_special_tokens:
        print(f"{token:15} → {label}")
```

**Filtering:** Skips special tokens ([CLS], [SEP], [PAD]) for cleaner output.

## Usage Examples

### Basic Execution

```bash
# Start server first
vllm serve boltuix/NeuroBERT-NER --runner pooling

# Run client
python ner_client.py
```

**Output:**
```
Barack          → B-PER
Obama           → I-PER
visited         → O
Microsoft       → B-ORG
headquarters    → O
in              → O
Seattle         → B-LOC
on              → O
January         → B-DATE
2025            → I-DATE
```

### Custom Server Configuration

```bash
python ner_client.py --host 192.168.1.100 --port 8080
```

**Use Cases:**
- Remote server access
- Load balancer endpoints
- Custom port configurations

### Different NER Model

```bash
# Server
vllm serve dslim/bert-base-NER --runner pooling

# Client
python ner_client.py --model dslim/bert-base-NER
```

**Model Consistency:** Client and server must use the same model for tokenization compatibility.

## API Specification

### Request Format

```json
{
  "model": "string (required)",
  "input": "string (required)",
  "encoding_format": "string (optional)"
}
```

**Parameters:**
- `model`: Model identifier (must match server)
- `input`: Text to analyze
- `encoding_format`: Not used for token classification

### Response Format

```json
{
  "id": "string",
  "object": "list",
  "created": integer,
  "model": "string",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "data": [
        [logit_11, logit_12, ..., logit_1N],
        [logit_21, logit_22, ..., logit_2N],
        ...
      ]
    }
  ],
  "usage": {
    "prompt_tokens": integer,
    "total_tokens": integer
  }
}
```

**Key Fields:**
- `data[0].data`: 2D array of logits `[seq_len, num_labels]`
- `usage.prompt_tokens`: Token count (verify alignment)

## Integration Patterns

### Batch Processing

```python
def process_documents(documents, api_url, model_name, tokenizer, label_map):
    """Process multiple documents."""
    all_entities = []

    for doc in documents:
        # Send request
        prompt = {"model": model_name, "input": doc}
        response = post_http_request(prompt, api_url)

        # Extract entities
        entities = extract_entities_from_response(
            response, doc, tokenizer, label_map
        )
        all_entities.append(entities)

    return all_entities
```

### Entity Extraction Wrapper

```python
def extract_entities_from_response(response, text, tokenizer, label_map):
    """Extract structured entities from API response."""
    output = response.json()["data"][0]
    logits = torch.tensor(output["data"])
    predictions = logits.argmax(dim=-1)

    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_map[p.item()] for p in predictions]

    # Build entity list
    entities = []
    current_entity = None

    for token, label in zip(tokens, labels):
        if token in tokenizer.all_special_tokens:
            continue

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": label[2:],
                "text": token.replace("##", "")
            }
        elif label.startswith("I-") and current_entity:
            current_entity["text"] += token.replace("##", "")

    if current_entity:
        entities.append(current_entity)

    return entities
```

**Output Example:**
```python
[
    {"type": "PER", "text": "BarackObama"},
    {"type": "ORG", "text": "Microsoft"},
    {"type": "LOC", "text": "Seattle"},
    {"type": "DATE", "text": "January2025"}
]
```

### Production Service Pattern

```python
class NERService:
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.label_map = self.config.id2label

    def extract_entities(self, text):
        """Extract entities from text."""
        prompt = {"model": self.model_name, "input": text}
        response = post_http_request(prompt, self.api_url)

        return self._process_response(response, text)

    def _process_response(self, response, text):
        """Process API response into entities."""
        # ... entity extraction logic ...
        return entities

# Usage
ner = NERService("http://localhost:8000/pooling", "boltuix/NeuroBERT-NER")
entities = ner.extract_entities("Some text...")
```

## Server Requirements

### Starting the Server

```bash
vllm serve boltuix/NeuroBERT-NER --runner pooling
```

**Required Flags:**
- `--runner pooling`: Enables token classification endpoint
- Model must have token classification head

### Server Configuration

```bash
vllm serve boltuix/NeuroBERT-NER \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.8
```

**Performance Options:**
- `--host 0.0.0.0`: Allow external connections
- `--max-model-len`: Context window limit
- `--gpu-memory-utilization`: Memory allocation

## Error Handling

### Connection Errors

```python
try:
    response = post_http_request(prompt, api_url)
    response.raise_for_status()
except requests.exceptions.ConnectionError:
    print("Server not reachable. Is vLLM server running?")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
```

### Token Alignment Errors

```python
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
labels = [label_map[p.item()] for p in predictions]

if len(tokens) != len(predictions):
    raise ValueError(
        f"Token count mismatch: {len(tokens)} tokens vs "
        f"{len(predictions)} predictions. "
        f"Ensure client and server use same model/tokenizer."
    )
```

### Label Map Errors

```python
try:
    label_map = config.id2label
except AttributeError:
    print("Model config missing id2label. Not a token classification model?")
    sys.exit(1)
```

## Performance Considerations

### Latency Components

```
Total Latency = Network + Tokenization + Inference + Decoding

Network:      10-50ms   (client-server round trip)
Tokenization: 1-5ms     (server-side)
Inference:    10-100ms  (model forward pass)
Decoding:     1-5ms     (client-side processing)
```

### Throughput Optimization

**Server-Side Batching:**
```bash
# Enable continuous batching on server
vllm serve boltuix/NeuroBERT-NER \
  --runner pooling \
  --max-num-batched-tokens 8192
```

**Client-Side Connection Pooling:**
```python
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.1)
session.mount('http://', HTTPAdapter(max_retries=retries))

response = session.post(api_url, json=prompt)
```

## Comparison with Offline Mode

| Aspect | Client (Online) | Offline |
|--------|----------------|---------|
| **Deployment** | Server + Client | Single process |
| **Scaling** | Horizontal | Vertical only |
| **Latency** | +Network overhead | Lower |
| **Resource** | Centralized GPU | Local GPU required |
| **Use Case** | Production services | Data pipelines |

## Dependencies

```python
import argparse           # CLI argument parsing
import requests           # HTTP client
import torch              # Tensor operations
from transformers import  # Model config and tokenizer
    AutoConfig,
    AutoTokenizer
```

**Installation:**
```bash
pip install requests torch transformers
```

## Best Practices

1. **Model Consistency**: Ensure client and server use identical models
2. **Token Alignment**: Always verify token count matches prediction count
3. **Connection Pooling**: Reuse HTTP connections for multiple requests
4. **Error Handling**: Implement retries and fallback logic
5. **Caching**: Cache tokenizer and config to avoid repeated downloads
6. **Async Requests**: Use async for high-throughput scenarios

## Related Components

- **NER Offline Example**: Local inference without server
- **Multi-Vector Retrieval Client**: Similar pooling endpoint usage
- **Token Classification Models**: Compatible NER model architectures
- **Pooling Runner**: vLLM server component

## References

- **Source File**: `examples/pooling/token_classify/ner_client.py`
- **API Endpoint**: `/pooling`
- **Server Command**: `vllm serve <model> --runner pooling`
- **Model**: boltuix/NeuroBERT-NER
- **Adapted From**: https://huggingface.co/boltuix/NeuroBERT-NER
- **Repository**: https://github.com/vllm-project/vllm

---

*This client demonstrates online named entity recognition using vLLM's token classification API for scalable NER services.*
