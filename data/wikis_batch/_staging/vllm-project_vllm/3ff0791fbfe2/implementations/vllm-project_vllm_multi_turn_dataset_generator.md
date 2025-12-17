---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/multi_turn/bench_dataset.py
domains:
  - Dataset Generation
  - Multi-turn Conversations
  - Benchmarking
  - Statistical Distributions
last_updated: 2025-12-17
---

# Multi-Turn Conversation Dataset Generator

## Overview

Comprehensive synthetic dataset generator for creating realistic multi-turn conversation workloads with configurable statistical distributions for benchmarking conversational AI systems.

## Description

The `bench_dataset.py` module provides a sophisticated framework for generating synthetic multi-turn conversation datasets that simulate realistic chat patterns. It supports multiple statistical distributions to model various conversation characteristics:

**Distribution Classes:**
- **UniformDistribution**: Uniform random values between min/max
- **ConstantDistribution**: Fixed constant values
- **ZipfDistribution**: Power-law distribution (common in natural language)
- **PoissonDistribution**: Discrete event distribution
- **LognormalDistribution**: Skewed distribution with configurable mean/median

**Configurable Parameters:**
- `num_turns`: Number of conversation turns (user + assistant pairs)
- `num_tokens`: Token counts for user prompts and assistant responses
- `prefix_num_tokens`: Context/history tokens added to first turn
- `common_prefix_num_tokens`: Shared prefix across all conversations

The generator creates conversations from source text files, ensuring:
- Unique conversation IDs to avoid caching effects
- Realistic token distributions matching production patterns
- Proper turn alternation (user → assistant → user...)
- Statistical validation and reporting

## Usage

### Importing Classes

```python
from benchmarks.multi_turn.bench_dataset import (
    Distribution,
    UniformDistribution,
    ConstantDistribution,
    ZipfDistribution,
    PoissonDistribution,
    LognormalDistribution,
    GenConvArgs,
    generate_conversations,
    parse_input_json_file,
    print_conv_stats,
)
```

### JSON Configuration File

```json
{
  "filetype": "generate_conversations",
  "num_conversations": 1000,
  "text_files": ["corpus1.txt", "corpus2.txt"],
  "prompt_input": {
    "num_turns": {
      "distribution": "zipf",
      "alpha": 1.5,
      "max": 20
    },
    "common_prefix_num_tokens": {
      "distribution": "constant",
      "value": 100
    },
    "prefix_num_tokens": {
      "distribution": "lognormal",
      "average": 500,
      "median_ratio": 0.85,
      "max": 2000
    },
    "num_tokens": {
      "distribution": "lognormal",
      "average": 150,
      "median_ratio": 0.8,
      "max": 512
    }
  },
  "prompt_output": {
    "num_tokens": {
      "distribution": "lognormal",
      "average": 200,
      "median_ratio": 0.75,
      "max": 1024
    }
  },
  "print_stats": true
}
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/multi_turn/bench_dataset.py`

**Key Classes:**

```python
class Distribution(ABC):
    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        pass

class LognormalDistribution(Distribution):
    def __init__(
        self,
        mean: float | None = None,
        sigma: float | None = None,
        average: int | None = None,
        median_ratio: float | None = None,
        max_val: int | None = None
    )

class GenConvArgs(NamedTuple):
    num_conversations: int
    text_files: list[str]
    input_num_turns: Distribution
    input_common_prefix_num_tokens: Distribution
    input_prefix_num_tokens: Distribution
    input_num_tokens: Distribution
    output_num_tokens: Distribution
    print_stats: bool
```

**Main Functions:**

```python
def generate_conversations(
    args: GenConvArgs,
    tokenizer: AutoTokenizer
) -> ConversationsMap

def parse_input_json_file(conf: dict) -> GenConvArgs

def print_conv_stats(
    conversations: ConversationsMap,
    tokenizer: AutoTokenizer
) -> None
```

**Import:**
```python
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_conversations` | `int` | Number of conversations to generate |
| `text_files` | `list[str]` | Source text files for content |
| `input_num_turns` | `Distribution` | Turn count distribution |
| `input_common_prefix_num_tokens` | `Distribution` | Shared prefix tokens |
| `input_prefix_num_tokens` | `Distribution` | Per-conversation prefix tokens |
| `input_num_tokens` | `Distribution` | User prompt token distribution |
| `output_num_tokens` | `Distribution` | Assistant response token distribution |
| `tokenizer` | `AutoTokenizer` | Tokenizer for text encoding |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `conversations` | `ConversationsMap` | Dict mapping conversation ID to message list |

**ConversationsMap Structure:**

```python
{
    "CONV_ID_0": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "CONV_ID_1": [...],
}
```

### Statistics Output

When `print_stats=True`, prints:

```
Conversations statistics:
              count    mean     std   min   25%   50%   75%   90%   99%    max
conversation_turns
user_tokens
assistant_tokens

Request statistics:
              count    mean     std   min   25%   50%   75%   90%   99%    max
request_tokens
```

## Usage Examples

### Basic Dataset Generation

```python
from benchmarks.multi_turn.bench_dataset import (
    GenConvArgs,
    UniformDistribution,
    LognormalDistribution,
    generate_conversations,
)
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

# Configure generation
args = GenConvArgs(
    num_conversations=100,
    text_files=["data/corpus.txt"],
    input_num_turns=UniformDistribution(2, 10),
    input_common_prefix_num_tokens=ConstantDistribution(0),
    input_prefix_num_tokens=UniformDistribution(100, 500),
    input_num_tokens=LognormalDistribution(average=150, max_val=512),
    output_num_tokens=LognormalDistribution(average=200, max_val=1024),
    print_stats=True
)

# Generate conversations
conversations = generate_conversations(args, tokenizer)
print(f"Generated {len(conversations)} conversations")
```

### Using JSON Configuration

```python
import json
from benchmarks.multi_turn.bench_dataset import parse_input_json_file, generate_conversations
from transformers import AutoTokenizer

# Load configuration
with open("config.json") as f:
    config = json.load(f)

# Parse config
args = parse_input_json_file(config)

# Generate dataset
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")
conversations = generate_conversations(args, tokenizer)

# Save to file
import json
from bench_dataset import conversations_dict_to_list

output = conversations_dict_to_list(conversations)
with open("conversations.json", "w") as f:
    json.dump(output, f, indent=2)
```

### Simulating Zipf Distribution (Realistic Chat Patterns)

```python
from benchmarks.multi_turn.bench_dataset import (
    ZipfDistribution,
    LognormalDistribution,
    generate_conversations,
    GenConvArgs,
)
from transformers import AutoTokenizer

# Zipf distribution models real-world patterns:
# - Most conversations are short (few turns)
# - Few conversations are very long
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

args = GenConvArgs(
    num_conversations=1000,
    text_files=["data/corpus1.txt", "data/corpus2.txt"],
    input_num_turns=ZipfDistribution(alpha=1.5, max_val=20),
    input_common_prefix_num_tokens=ConstantDistribution(0),
    input_prefix_num_tokens=LognormalDistribution(average=300, max_val=1000),
    input_num_tokens=LognormalDistribution(average=100, median_ratio=0.8, max_val=512),
    output_num_tokens=LognormalDistribution(average=150, median_ratio=0.75, max_val=1024),
    print_stats=True
)

conversations = generate_conversations(args, tokenizer)
```

### Testing Prefix Caching Scenarios

```python
from benchmarks.multi_turn.bench_dataset import (
    ConstantDistribution,
    LognormalDistribution,
    GenConvArgs,
    generate_conversations,
)
from transformers import AutoTokenizer

# Generate conversations with large shared prefix
# (tests prefix caching effectiveness)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

args = GenConvArgs(
    num_conversations=500,
    text_files=["data/corpus.txt"],
    input_num_turns=UniformDistribution(4, 8),
    # Large shared prefix across all conversations
    input_common_prefix_num_tokens=ConstantDistribution(1000),
    # Medium per-conversation context
    input_prefix_num_tokens=LognormalDistribution(average=500, max_val=1500),
    input_num_tokens=LognormalDistribution(average=100, max_val=256),
    output_num_tokens=LognormalDistribution(average=150, max_val=512),
    print_stats=True
)

conversations = generate_conversations(args, tokenizer)
print("Dataset optimized for prefix caching benchmarks")
```

### Analyzing Generated Dataset

```python
from benchmarks.multi_turn.bench_dataset import (
    generate_conversations,
    print_conv_stats,
)
from transformers import AutoTokenizer
import pandas as pd

# Generate dataset
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")
conversations = generate_conversations(args, tokenizer)

# Print built-in statistics
print_conv_stats(conversations, tokenizer)

# Custom analysis
turn_counts = []
total_tokens = []

for conv_id, messages in conversations.items():
    turn_counts.append(len(messages))

    # Calculate total tokens in conversation
    tokens = 0
    for msg in messages:
        tokens += len(tokenizer(msg["content"]).input_ids)
    total_tokens.append(tokens)

# Create DataFrame
df = pd.DataFrame({
    "turns": turn_counts,
    "total_tokens": total_tokens
})

print("\nCustom Statistics:")
print(df.describe())

# Plot distributions
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(turn_counts, bins=20)
ax1.set_xlabel("Number of Turns")
ax1.set_ylabel("Frequency")
ax1.set_title("Turn Count Distribution")

ax2.hist(total_tokens, bins=20)
ax2.set_xlabel("Total Tokens")
ax2.set_ylabel("Frequency")
ax2.set_title("Token Count Distribution")

plt.tight_layout()
plt.show()
```

### Creating Custom Distribution

```python
from benchmarks.multi_turn.bench_dataset import Distribution
import numpy as np

class BimodalDistribution(Distribution):
    """Custom bimodal distribution (short and long conversations)"""

    def __init__(self, short_mean=3, long_mean=15, short_weight=0.7):
        self.short_mean = short_mean
        self.long_mean = long_mean
        self.short_weight = short_weight

    def sample(self, size: int = 1) -> np.ndarray:
        # Sample from two populations
        is_short = np.random.random(size) < self.short_weight

        samples = np.zeros(size, dtype=int)
        samples[is_short] = np.random.poisson(self.short_mean, is_short.sum())
        samples[~is_short] = np.random.poisson(self.long_mean, (~is_short).sum())

        # Ensure even number of turns (user + assistant pairs)
        samples = samples + (samples % 2)

        return np.maximum(samples, 2)  # At least 2 turns

# Use custom distribution
from benchmarks.multi_turn.bench_dataset import GenConvArgs, generate_conversations
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

args = GenConvArgs(
    num_conversations=200,
    text_files=["data/corpus.txt"],
    input_num_turns=BimodalDistribution(short_mean=3, long_mean=20),
    input_common_prefix_num_tokens=ConstantDistribution(0),
    input_prefix_num_tokens=UniformDistribution(100, 500),
    input_num_tokens=LognormalDistribution(average=120, max_val=512),
    output_num_tokens=LognormalDistribution(average=180, max_val=1024),
    print_stats=True
)

conversations = generate_conversations(args, tokenizer)
```

### Validating Lognormal Parameters

```python
from benchmarks.multi_turn.bench_dataset import LognormalDistribution
import numpy as np
import matplotlib.pyplot as plt

# Test lognormal with target average
dist = LognormalDistribution(average=150, median_ratio=0.8, max_val=512)

# Generate large sample
samples = dist.sample(10000)

print(f"Target average: 150")
print(f"Actual average: {samples.mean():.1f}")
print(f"Median: {np.median(samples):.1f}")
print(f"Median/Mean ratio: {np.median(samples) / samples.mean():.2f}")
print(f"Min: {samples.min()}")
print(f"Max: {samples.max()}")

# Plot histogram
plt.hist(samples, bins=50, density=True, alpha=0.7)
plt.axvline(samples.mean(), color='r', linestyle='--', label='Mean')
plt.axvline(np.median(samples), color='g', linestyle='--', label='Median')
plt.xlabel('Token Count')
plt.ylabel('Density')
plt.title('Lognormal Distribution (average=150, median_ratio=0.8)')
plt.legend()
plt.show()
```

### Converting Between Formats

```python
from benchmarks.multi_turn.bench_dataset import (
    conversations_list_to_dict,
    conversations_dict_to_list,
)
import json

# Load ShareGPT-style data
with open("sharegpt_data.json") as f:
    sharegpt_list = json.load(f)

# Convert to dict format for processing
conversations_dict = conversations_list_to_dict(sharegpt_list)

# Process...
# (filter, modify, etc.)

# Convert back to list format for saving
output_list = conversations_dict_to_list(conversations_dict)

with open("processed_conversations.json", "w") as f:
    json.dump(output_list, f, indent=2)
```

## Related Pages

- [[vllm-project_vllm_sharegpt_converter]] - ShareGPT format converter
- [[vllm-project_vllm_multi_turn_benchmarks]] - Multi-turn benchmarking suite
- [[vllm-project_vllm_prefix_caching]] - Prefix caching implementation
- [[vllm-project_vllm_conversation_format]] - Conversation format specifications
- [[vllm-project_vllm_openai_chat_completion]] - OpenAI chat completion API
- [[vllm-project_vllm_stateful_serving]] - Stateful serving for conversations
