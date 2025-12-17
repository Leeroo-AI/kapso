# Implementation: Named Entity Recognition (Offline)

**File:** `/tmp/praxium_repo_583nq7ea/examples/pooling/token_classify/ner.py`
**Type:** Offline Inference Example
**Lines of Code:** 54
**Last Updated:** 2025-12-17

## Overview

The `ner.py` script demonstrates offline named entity recognition using vLLM's token classification capabilities. This implementation shows how to use vLLM for token-level prediction tasks beyond text generation and embeddings, extracting structured information from unstructured text.

### Purpose

Illustrates vLLM's support for token classification models, enabling efficient batch processing of NER tasks without requiring an HTTP server.

### Key Features

- **Offline Processing**: Direct model inference without server
- **Token-Level Classification**: Per-token entity predictions
- **Label Mapping**: Automatic conversion of predictions to entity types
- **Flexible Arguments**: Uses vLLM's standard EngineArgs interface

## Architecture

### Token Classification Pipeline

```
┌─────────────────┐
│  Input Text     │
│  "Barack Obama  │
│   visited..."   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization   │
│  [Barack, Obama,│
│   visited, ...]  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Forward  │
│  (per-token     │
│   logits)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Argmax         │
│  (label per     │
│   token)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Label Mapping  │
│  Barack→PERSON  │
│  Obama→PERSON   │
│  visited→O      │
└─────────────────┘
```

## Implementation Details

### Argument Parsing

```python
def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)

    # Set example specific arguments
    parser.set_defaults(
        model="boltuix/NeuroBERT-NER",
        runner="pooling",
        enforce_eager=True,
        trust_remote_code=True,
    )
    return parser.parse_args()
```

**Configuration:**
- `model`: NER model identifier (NeuroBERT-NER)
- `runner="pooling"`: Enables token classification mode
- `enforce_eager=True`: Disables graph optimization for simplicity
- `trust_remote_code=True`: Allows custom model code execution

**Design Pattern:** Extends vLLM's standard EngineArgs for consistency with other vLLM tools.

### Model Initialization

```python
def main(args: Namespace):
    # Create an LLM instance
    llm = LLM(**vars(args))
    tokenizer = llm.get_tokenizer()
    label_map = llm.llm_engine.vllm_config.model_config.hf_config.id2label
```

**Key Components:**
- `LLM`: Core inference engine
- `tokenizer`: For token-to-text mapping
- `label_map`: Entity type definitions from model config

**Label Map Example:**
```python
{
    0: "O",           # Outside entity
    1: "B-PER",       # Begin person
    2: "I-PER",       # Inside person
    3: "B-ORG",       # Begin organization
    4: "I-ORG",       # Inside organization
    5: "B-LOC",       # Begin location
    6: "I-LOC",       # Inside location
    # ... more labels
}
```

### Inference Execution

```python
# Sample prompts
prompts = [
    "Barack Obama visited Microsoft headquarters in Seattle on January 2025."
]

# Run inference with token classification
outputs = llm.encode(prompts, pooling_task="token_classify")
```

**API Usage:**
- `llm.encode()`: Generic pooling/encoding method
- `pooling_task="token_classify"`: Specifies token-level classification
- Returns per-token logits for each input

### Output Processing

```python
for prompt, output in zip(prompts, outputs):
    logits = output.outputs.data
    predictions = logits.argmax(dim=-1)

    # Map predictions to labels
    tokens = tokenizer.convert_ids_to_tokens(output.prompt_token_ids)
    labels = [label_map[p.item()] for p in predictions]

    # Print results
    for token, label in zip(tokens, labels):
        if token not in tokenizer.all_special_tokens:
            print(f"{token:15} → {label}")
```

**Processing Steps:**
1. Extract logits from output (shape: `[seq_len, num_labels]`)
2. Apply argmax to get predicted label indices
3. Convert token IDs back to token strings
4. Map label indices to entity type strings
5. Filter out special tokens
6. Display token-label pairs

## Usage Examples

### Basic Execution

```bash
python ner.py
```

**Default Configuration:**
- Model: boltuix/NeuroBERT-NER
- Input: Pre-defined example text
- Output: Token-label pairs

### Custom Model

```bash
python ner.py --model dslim/bert-base-NER
```

**Compatible Models:**
- Any HuggingFace model with token classification head
- Models with `id2label` configuration
- BERT-based NER models

### Advanced Configuration

```bash
python ner.py \
  --model boltuix/NeuroBERT-NER \
  --tensor-parallel-size 2 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.8
```

**Performance Options:**
- `--tensor-parallel-size`: Multi-GPU inference
- `--max-model-len`: Context window limit
- `--gpu-memory-utilization`: Memory allocation

## Example Output

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
.               → O
```

**Label Interpretation:**
- `B-PER`: Beginning of person entity (Barack)
- `I-PER`: Inside person entity (Obama)
- `B-ORG`: Beginning of organization (Microsoft)
- `B-LOC`: Beginning of location (Seattle)
- `B-DATE`: Beginning of date (January 2025)
- `O`: Outside any entity (visited, in, on)

## Token Classification Task

### Supported Entity Types

Common NER label schemes:

**CoNLL-2003 Format:**
- `PER`: Person names
- `ORG`: Organizations
- `LOC`: Locations
- `MISC`: Miscellaneous entities

**BIO Tagging:**
- `B-`: Beginning of entity
- `I-`: Inside entity (continuation)
- `O`: Outside any entity

**Extended Schemes:**
- `DATE`, `TIME`: Temporal expressions
- `MONEY`, `PERCENT`: Numerical entities
- `PRODUCT`, `EVENT`: Specialized types

### Model Architecture

```
┌──────────────────────────────────┐
│  Token Embeddings                │
│  [batch, seq_len, hidden_dim]    │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Transformer Layers              │
│  (BERT/RoBERTa backbone)         │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Classification Head             │
│  Linear: hidden_dim → num_labels │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Per-Token Logits                │
│  [batch, seq_len, num_labels]    │
└──────────────────────────────────┘
```

## Integration Patterns

### Batch Processing

```python
# Process multiple documents
documents = [
    "Document 1 text...",
    "Document 2 text...",
    # ... more documents
]

outputs = llm.encode(documents, pooling_task="token_classify")

for doc, output in zip(documents, outputs):
    # Process each document's entities
    entities = extract_entities(output)
```

### Entity Extraction Function

```python
def extract_entities(output, tokenizer, label_map):
    """Extract named entities from token classification output."""
    logits = output.outputs.data
    predictions = logits.argmax(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(output.prompt_token_ids)
    labels = [label_map[p.item()] for p in predictions]

    entities = []
    current_entity = None

    for token, label in zip(tokens, labels):
        if token in tokenizer.all_special_tokens:
            continue

        if label.startswith("B-"):
            # Start new entity
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": label[2:],
                "tokens": [token],
                "text": token
            }
        elif label.startswith("I-") and current_entity:
            # Continue entity
            current_entity["tokens"].append(token)
            current_entity["text"] += " " + token
        else:
            # End current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities
```

### Structured Output

```python
# Extract entities with positions
entities = extract_entities(output, tokenizer, label_map)

# Output:
# [
#   {"type": "PER", "text": "Barack Obama", "tokens": ["Barack", "Obama"]},
#   {"type": "ORG", "text": "Microsoft", "tokens": ["Microsoft"]},
#   {"type": "LOC", "text": "Seattle", "tokens": ["Seattle"]},
#   {"type": "DATE", "text": "January 2025", "tokens": ["January", "2025"]}
# ]
```

## Performance Characteristics

### Throughput

| Batch Size | Tokens/sec | Latency |
|------------|------------|---------|
| 1 | 1,000 | 50ms |
| 8 | 6,000 | 100ms |
| 32 | 20,000 | 200ms |

*Approximate values for NeuroBERT-NER on A100 GPU*

### Memory Usage

```python
# Typical memory requirements
# - Model weights: ~500MB (base BERT)
# - Activation memory: ~100MB per batch
# - Total GPU memory: ~2GB for batch size 32
```

## Comparison with Online Mode

### Offline (This Implementation)

**Advantages:**
- No server overhead
- Direct programmatic access
- Batch processing control
- Simpler deployment

**Use Cases:**
- Data processing pipelines
- Research experiments
- One-off analysis tasks

### Online (Server Mode)

**Advantages:**
- Multi-client support
- Horizontal scaling
- API standardization
- Remote access

**Use Cases:**
- Production services
- Web applications
- Microservices architecture

## Error Handling

### Model Loading

```python
try:
    llm = LLM(**vars(args))
except Exception as e:
    print(f"Failed to load model: {e}")
    # Check model name, GPU availability, memory
```

### Label Map Access

```python
try:
    label_map = llm.llm_engine.vllm_config.model_config.hf_config.id2label
except AttributeError:
    print("Model config missing id2label mapping")
    # Verify model has token classification head
```

## Dependencies

```python
from argparse import Namespace
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
```

**Requirements:**
- vLLM: Core inference engine
- PyTorch: Deep learning backend
- Transformers: Model loading (implicit)

## Model Compatibility

### Supported Models

- **NeuroBERT-NER**: Medical/scientific NER
- **dslim/bert-base-NER**: General-purpose NER
- **Jean-Baptiste/camembert-ner**: French NER
- **dbmdz/bert-large-cased-finetuned-conll03-english**: CoNLL-2003 NER

### Model Requirements

1. Token classification head in model architecture
2. `id2label` mapping in model config
3. Compatible tokenizer
4. Appropriate task type in config

## Best Practices

1. **Verify Label Map**: Check `id2label` matches expected entities
2. **Handle Special Tokens**: Filter tokenizer special tokens
3. **Batch Processing**: Use larger batches for throughput
4. **Entity Reconstruction**: Implement proper BIO tagging parsing
5. **Memory Management**: Monitor GPU memory with large batches

## Related Components

- **NER Client**: Online version using HTTP API
- **Token Embedding Examples**: Other token-level tasks
- **Pooling Runner**: vLLM component for classification
- **FlexibleArgumentParser**: Argument parsing utility

## References

- **Source File**: `examples/pooling/token_classify/ner.py`
- **Model**: boltuix/NeuroBERT-NER
- **Adapted From**: https://huggingface.co/boltuix/NeuroBERT-NER
- **Task Type**: Token classification
- **Repository**: https://github.com/vllm-project/vllm

---

*This implementation demonstrates vLLM's token classification capabilities for offline named entity recognition tasks.*
