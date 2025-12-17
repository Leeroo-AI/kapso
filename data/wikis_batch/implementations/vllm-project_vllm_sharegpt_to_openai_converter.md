---
knowledge_sources:
  - vllm-project/vllm repository
  - File: benchmarks/multi_turn/convert_sharegpt_to_openai.py
domains:
  - Dataset Conversion
  - Data Preprocessing
  - Multi-turn Conversations
  - Format Standardization
last_updated: 2025-12-17
---

# ShareGPT to OpenAI Format Converter

## Overview

Command-line tool for converting ShareGPT conversation datasets to OpenAI chat completion format with filtering, validation, and quality control features.

## Description

The `convert_sharegpt_to_openai.py` script provides comprehensive data preprocessing for transforming ShareGPT-format conversation datasets into the standardized OpenAI chat completion format used by vLLM's multi-turn benchmarking suite.

**Key Features:**
- **Format conversion**: ShareGPT → OpenAI chat completion format
- **Conversation merging**: Combines multi-part conversations by ID
- **Quality filtering**: Filters by content length, turn count, language
- **Validation**: Ensures proper turn alternation (user ↔ assistant)
- **Random sampling**: Controls output dataset size
- **Statistics reporting**: Token counts, turn distributions, and percentiles
- **Tokenizer integration**: Optional tokenizer for accurate token counting

**Filtering Options:**
- `--min-turns` / `--max-turns`: Filter by conversation length
- `--min-content-len` / `--max-content-len`: Filter by character count
- `--max-items`: Limit output dataset size via random sampling
- Non-English character detection for language filtering

**Input Format (ShareGPT):**
```json
[
  {
    "id": "conv123_0",
    "conversations": [
      {"from": "human", "value": "Hello!"},
      {"from": "gpt", "value": "Hi there!"}
    ]
  }
]
```

**Output Format (OpenAI):**
```json
[
  {
    "id": "conv123",
    "messages": [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there!"}
    ]
  }
]
```

## Usage

### Command Line Execution

```bash
# Basic conversion
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    input.json output.json

# Limit output size
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    input.json output.json --max-items=1000

# Filter by turn count
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    input.json output.json \
    --min-turns=4 --max-turns=20

# Filter by content length
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    input.json output.json \
    --min-content-len=10 --max-content-len=1000

# With tokenizer for accurate stats
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    input.json output.json \
    --model meta-llama/Llama-3-8b \
    --max-items=500

# Set random seed for reproducibility
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    input.json output.json \
    --seed=42 --max-items=1000
```

### Full Example

```bash
# Download ShareGPT dataset
wget https://huggingface.co/datasets/philschmid/sharegpt-raw/resolve/main/sharegpt_20230401_clean_lang_split.json

# Convert to OpenAI format (128 conversations)
export INPUT_FILE=sharegpt_20230401_clean_lang_split.json
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    $INPUT_FILE sharegpt_conv_128.json \
    --max-items=128 \
    --min-turns=2 \
    --max-turns=10
```

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/multi_turn/convert_sharegpt_to_openai.py`

**Main Function:**

```python
def convert_sharegpt_to_openai(
    seed: int,
    input_file: str,
    output_file: str,
    max_items: int | None,
    min_content_len: int | None = None,
    max_content_len: int | None = None,
    min_turns: int | None = None,
    max_turns: int | None = None,
    model: str | None = None
) -> None
```

**Helper Functions:**

```python
def has_non_english_chars(text: str) -> bool:
    """Check if text contains non-ASCII characters"""
    return not text.isascii()

def content_is_valid(
    content: str,
    min_content_len: int | None,
    max_content_len: int | None
) -> bool:
    """Validate content length and language"""

def print_stats(
    conversations: list[dict[Any, Any]],
    tokenizer: AutoTokenizer | None = None
) -> None:
    """Print conversation statistics"""
```

**Import:**
```python
from transformers import AutoTokenizer
import pandas as pd
import random
```

## I/O Contract

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | `str` | Path to ShareGPT JSON file |
| `output_file` | `str` | Path for output OpenAI JSON file |
| `seed` | `int` | Random seed (default: 0) |
| `max_items` | `int \| None` | Maximum conversations in output |
| `min_content_len` | `int \| None` | Minimum characters per message |
| `max_content_len` | `int \| None` | Maximum characters per message |
| `min_turns` | `int \| None` | Minimum turns per conversation |
| `max_turns` | `int \| None` | Maximum turns per conversation |
| `model` | `str \| None` | Model for tokenizer (optional) |

### ShareGPT Input Format

```python
[
  {
    "id": str,              # Conversation ID with part: "conv123_0", "conv123_1"
    "conversations": [
      {
        "from": str,        # "human", "gpt", "user", "bing", "chatgpt", "bard", "system"
        "value": str        # Message content
      }
    ]
  }
]
```

### OpenAI Output Format

```python
[
  {
    "id": str,              # Conversation ID (merged parts): "conv123"
    "messages": [
      {
        "role": str,        # "user", "assistant", "system"
        "content": str      # Message content
      }
    ]
  }
]
```

### Statistics Output

Prints to console:

```
Statistics:
                   count      mean       std    min     25%     50%     75%     90%     99%    max
user_turns         1000       3.5        2.1    1.0     2.0     3.0     5.0     6.0     10.0   12.0
assistant_turns    1000       3.5        2.1    1.0     2.0     3.0     5.0     6.0     10.0   12.0
user_words         1000     125.3       87.4   10.0    65.0   110.0   175.0   230.0   410.0  650.0
assistant_words    1000     178.6      132.5   15.0    85.0   145.0   245.0   350.0   625.0  980.0
conv_turns         1000       7.0        4.2    2.0     4.0     6.0    10.0    12.0    20.0   24.0
user_tokens        1000      95.2       65.8    8.0    50.0    85.0   135.0   180.0   315.0  495.0
assistant_tokens   1000     138.4      102.3   12.0    65.0   112.0   190.0   270.0   485.0  760.0
```

## Usage Examples

### Basic Conversion

```python
from benchmarks.multi_turn.convert_sharegpt_to_openai import convert_sharegpt_to_openai

convert_sharegpt_to_openai(
    seed=42,
    input_file="sharegpt_raw.json",
    output_file="openai_format.json",
    max_items=1000,
    min_content_len=10,
    max_content_len=2000,
    min_turns=2,
    max_turns=20,
    model=None  # No tokenizer
)
```

### With Tokenizer Statistics

```python
from benchmarks.multi_turn.convert_sharegpt_to_openai import convert_sharegpt_to_openai

# Convert with accurate token counting
convert_sharegpt_to_openai(
    seed=42,
    input_file="sharegpt_raw.json",
    output_file="openai_format.json",
    max_items=500,
    min_turns=3,
    max_turns=15,
    model="meta-llama/Llama-3-8b"  # Use Llama-3 tokenizer
)
```

### Quality Filtering

```python
from benchmarks.multi_turn.convert_sharegpt_to_openai import convert_sharegpt_to_openai

# Strict quality filters
convert_sharegpt_to_openai(
    seed=42,
    input_file="sharegpt_raw.json",
    output_file="high_quality.json",
    max_items=100,
    min_content_len=50,      # At least 50 chars per message
    max_content_len=1000,    # At most 1000 chars per message
    min_turns=4,             # Multi-turn only (4+ turns)
    max_turns=10,            # Not too long (≤10 turns)
    model="meta-llama/Llama-3-8b"
)

print("High-quality filtered dataset created")
```

### Processing Large Datasets

```python
import subprocess

# Process in batches
batch_sizes = [100, 500, 1000, 5000]

for batch_size in batch_sizes:
    output_file = f"sharegpt_openai_{batch_size}.json"

    subprocess.run([
        "python",
        "benchmarks/multi_turn/convert_sharegpt_to_openai.py",
        "sharegpt_raw.json",
        output_file,
        "--seed", "42",
        "--max-items", str(batch_size),
        "--min-turns", "2",
        "--max-turns", "20"
    ])

    print(f"Created {output_file} with {batch_size} conversations")
```

### Analyzing Converted Data

```python
import json
import pandas as pd

# Load converted data
with open("openai_format.json") as f:
    conversations = json.load(f)

# Analyze conversation characteristics
stats = []
for conv in conversations:
    messages = conv["messages"]

    user_msgs = [m for m in messages if m["role"] == "user"]
    asst_msgs = [m for m in messages if m["role"] == "assistant"]

    stats.append({
        "id": conv["id"],
        "total_turns": len(messages),
        "user_turns": len(user_msgs),
        "asst_turns": len(asst_msgs),
        "avg_user_len": sum(len(m["content"]) for m in user_msgs) / len(user_msgs),
        "avg_asst_len": sum(len(m["content"]) for m in asst_msgs) / len(asst_msgs),
    })

df = pd.DataFrame(stats)
print(df.describe())

# Check for imbalances
imbalanced = df[df["user_turns"] != df["asst_turns"]]
if len(imbalanced) > 0:
    print(f"\nWarning: {len(imbalanced)} conversations have imbalanced turns")
```

### Validating Output Format

```python
import json

def validate_openai_format(file_path):
    """Validate OpenAI format correctness"""
    with open(file_path) as f:
        conversations = json.load(f)

    issues = []

    for i, conv in enumerate(conversations):
        # Check required fields
        if "id" not in conv or "messages" not in conv:
            issues.append(f"Conv {i}: Missing 'id' or 'messages'")
            continue

        messages = conv["messages"]

        # Check turn alternation
        expected_role = "user"
        for j, msg in enumerate(messages):
            if msg["role"] != expected_role:
                issues.append(
                    f"Conv {i}, msg {j}: Expected {expected_role}, "
                    f"got {msg['role']}"
                )

            expected_role = "assistant" if expected_role == "user" else "user"

        # Check last turn
        if len(messages) > 0 and messages[-1]["role"] != "assistant":
            issues.append(f"Conv {i}: Last message should be from assistant")

    if issues:
        print(f"Found {len(issues)} validation issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
    else:
        print("All conversations valid!")

    return len(issues) == 0

# Validate
validate_openai_format("openai_format.json")
```

### Creating Test/Train Splits

```python
from benchmarks.multi_turn.convert_sharegpt_to_openai import convert_sharegpt_to_openai
import random
import json

# Convert full dataset
convert_sharegpt_to_openai(
    seed=42,
    input_file="sharegpt_raw.json",
    output_file="all_conversations.json",
    max_items=1000,
    min_turns=2,
    max_turns=20
)

# Load and split
with open("all_conversations.json") as f:
    all_convs = json.load(f)

random.seed(42)
random.shuffle(all_convs)

# 80/20 train/test split
split_idx = int(len(all_convs) * 0.8)
train_convs = all_convs[:split_idx]
test_convs = all_convs[split_idx:]

# Save splits
with open("train.json", "w") as f:
    json.dump(train_convs, f, indent=2)

with open("test.json", "w") as f:
    json.dump(test_convs, f, indent=2)

print(f"Train: {len(train_convs)} conversations")
print(f"Test: {len(test_convs)} conversations")
```

### Filtering by Topic/Content

```python
import json

def filter_by_keywords(input_file, output_file, keywords, exclude=False):
    """Filter conversations by keyword presence"""
    with open(input_file) as f:
        conversations = json.load(f)

    filtered = []
    for conv in conversations:
        # Check all messages for keywords
        has_keyword = False
        for msg in conv["messages"]:
            content_lower = msg["content"].lower()
            if any(kw.lower() in content_lower for kw in keywords):
                has_keyword = True
                break

        # Include or exclude based on flag
        if (has_keyword and not exclude) or (not has_keyword and exclude):
            filtered.append(conv)

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Filtered {len(filtered)}/{len(conversations)} conversations")

# Example: Filter for coding conversations
filter_by_keywords(
    "openai_format.json",
    "coding_conversations.json",
    keywords=["python", "code", "function", "programming", "debug"],
    exclude=False
)
```

## Related Pages

- [[vllm-project_vllm_multi_turn_dataset_generator]] - Synthetic dataset generation
- [[vllm-project_vllm_conversation_format]] - Conversation format specifications
- [[vllm-project_vllm_openai_chat_completion]] - OpenAI chat completion API
- [[vllm-project_vllm_multi_turn_benchmarks]] - Multi-turn benchmarking
- [[vllm-project_vllm_dataset_preprocessing]] - Dataset preprocessing utilities
- [[vllm-project_vllm_sharegpt_format]] - ShareGPT format documentation
