# src/peft/tuners/trainable_tokens/config.py

## Overview
Configuration class for the Trainable Tokens method, which allows training specific token embeddings while keeping the rest frozen. This configuration specifies which tokens should be trainable and how to initialize them.

## Class: TrainableTokensConfig

Inherits from `PeftConfig` and provides configuration for selective token training.

### Configuration Parameters

#### Token Selection
- **token_indices** (list[int], default=empty list):
  - List of token indices that should be trainable
  - Indices correspond to positions in the vocabulary
  - Find indices using tokenizer: `tokenizer("text")["input_ids"]`
  - **Performance Note**: The closer the number of indices to total vocabulary size, the less efficient this method becomes

**How to Find Token Indices:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model_name")
# Tokenize to get indices
tokens = tokenizer("Hello world", add_special_tokens=False)
print(tokens["input_ids"])  # e.g., [15496, 995]

# For single tokens
token_id = tokenizer.convert_tokens_to_ids("<special_token>")
```

#### Target Module Selection
- **target_modules** (Optional[Union[list[str], str]], default=None):
  - List of module names or regex patterns for embedding layers
  - If not defined: Attempts to use model's `get_input_embeddings()` method
  - Fallback default: `'embed_tokens'`
  - **Common alternatives**: `'embedding'`, `'encoder.embeddings'`, `'decoder.embeddings'`

**Automatic Detection:**
Most transformer models have `get_input_embeddings()` method, so target_modules can usually be left as None. The system will:
1. Try `model.get_input_embeddings()`
2. If that fails, use default `'embed_tokens'`
3. If neither works, raise error

**Manual Specification:**
Needed when:
- Model doesn't have `get_input_embeddings()`
- Want to target specific embedding layers
- Want to target multiple embedding layers

#### Weight Initialization
- **init_weights** (bool, default=True):
  - If True: Initialize trainable token embeddings from current embeddings (recommended)
  - If False: Initialize with random values
  - **Warning**: Changing from True is discouraged

**Why True is Recommended:**
- Makes Trainable Tokens a no-op when not trained
- Preserves existing token semantics
- Provides stable starting point
- Random initialization can disrupt model behavior

**When False Might Be Used:**
- Adding completely new tokens with no prior representation
- Intentionally resetting token embeddings
- Experimental scenarios only

### Post-Initialization

The `__post_init__` method:
1. Calls parent's post-init
2. Sets `peft_type` to `PeftType.TRAINABLE_TOKENS`

---

## Design Philosophy

### Selective Training
Rather than training entire embedding matrix:
- Identify specific tokens needing updates
- Only those tokens' embeddings are trainable
- Rest of embedding matrix remains frozen
- Significant memory savings

### Memory Efficiency
**Storage:**
- Only save updated token embeddings
- Checkpoint size = num_tokens * embedding_dim
- Full embedding: vocab_size * embedding_dim

**Working Memory:**
- Gradients only computed for selected tokens
- Optimizer states only for trainable tokens
- Significant reduction for large vocabularies

**Example:**
- Vocabulary: 50,000 tokens
- Embedding dimension: 768
- Full embedding: 38.4M parameters
- Training 100 tokens: 76.8K parameters
- Memory reduction: ~500x

### Efficiency Threshold
From the docstring: "The closer the amount of indices is to the total amount of tokens, the less efficient this method gets."

**Guidelines:**
- **<1% of vocab**: Very efficient
- **1-10% of vocab**: Still beneficial
- **>50% of vocab**: Consider full fine-tuning instead
- At high percentages, overhead may exceed benefits

---

## Use Case Patterns

### Adding New Vocabulary
```python
# Add 10 new domain-specific tokens
tokenizer.add_tokens(["<term1>", "<term2>", ...])
model.resize_token_embeddings(len(tokenizer))

# Get indices of new tokens
new_indices = [
    tokenizer.convert_tokens_to_ids(f"<term{i}>")
    for i in range(1, 11)
]

config = TrainableTokensConfig(
    token_indices=new_indices,
    init_weights=False,  # New tokens, no prior embedding
)
```

### Fine-tuning Existing Tokens
```python
# Improve embeddings for specific tokens
tokens_to_improve = ["important", "keyword", "domain-term"]
indices = [tokenizer.convert_tokens_to_ids(t) for t in tokens_to_improve]

config = TrainableTokensConfig(
    token_indices=indices,
    init_weights=True,  # Start from current embeddings
)
```

### Domain Adaptation
```python
# Adapt embeddings for domain-specific usage
# e.g., medical terms, legal terms, etc.
domain_tokens = ["diagnosis", "treatment", "symptom", ...]
indices = [tokenizer.convert_tokens_to_ids(t) for t in domain_tokens]

config = TrainableTokensConfig(
    token_indices=indices,
    target_modules=["embed_tokens"],  # Explicit targeting
)
```

### Multi-Embedding Targeting
```python
# Target both encoder and decoder embeddings
config = TrainableTokensConfig(
    token_indices=new_token_indices,
    target_modules=["encoder.embeddings", "decoder.embeddings"],
)
```

---

## Important Considerations

### Weight Tying
Models often tie embeddings across different layers:
- Input embeddings tied with output (LM head)
- Encoder embeddings tied with decoder
- TrainableTokensModel automatically handles this
- Updates propagate to all tied locations

### Tokenizer Changes
When adding new tokens:
1. Add tokens to tokenizer first
2. Resize model embeddings
3. Get indices of new tokens
4. Configure TrainableTokensConfig with indices
5. Apply PEFT

Order matters! Resize embeddings before applying PEFT.

### Index Validation
No automatic validation of indices. Ensure:
- Indices are within vocabulary range [0, vocab_size)
- Indices correspond to intended tokens
- No duplicates (though not harmful, just inefficient)

### Performance Trade-offs
**When to Use:**
- Small number of tokens (<1% of vocab)
- Memory constraints
- Adding new vocabulary
- Domain adaptation with focused changes

**When Not to Use:**
- Need to update >50% of vocabulary
- All tokens need adaptation
- Have sufficient memory for full fine-tuning
- Need more complex embedding transformations

---

## Integration
This configuration is passed to `get_peft_model()` or used directly with `TrainableTokensModel` to enable selective token training.

## Example Usage
```python
from peft import TrainableTokensConfig, get_peft_model

# Basic configuration
config = TrainableTokensConfig(
    task_type="CAUSAL_LM",
    token_indices=[100, 200, 300],  # Specific tokens to train
    init_weights=True,
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()  # Shows only selected tokens trainable
```
