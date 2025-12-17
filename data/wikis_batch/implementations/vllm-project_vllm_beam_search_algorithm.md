# Implementation: Beam Search Algorithm

**File:** `/tmp/praxium_repo_583nq7ea/vllm/beam_search.py`
**Type:** Core Algorithm Module
**Lines of Code:** 88
**Last Updated:** 2025-12-17

## Overview

The `beam_search.py` module implements beam search decoding for text generation in vLLM. This alternative to greedy and sampling-based decoding explores multiple hypotheses in parallel, enabling higher-quality outputs for tasks like translation, summarization, and structured generation where exploring alternatives improves results.

### Purpose

Provides data structures and algorithms for beam search decoding, allowing vLLM to generate multiple candidate sequences and select the best ones based on cumulative log probabilities with length normalization.

### Key Features

- **Beam State Management**: Tracks active and completed sequences
- **Length Normalization**: Prevents bias toward shorter sequences
- **Scoring Function**: Computes beam search scores with penalties
- **Path Tracking**: Maintains parent-child relationships between beams
- **Multi-Modal Support**: Handles text and multi-modal inputs

## Architecture

### Core Components

```
┌─────────────────────────────────────────────┐
│        BeamSearchInstance                   │
│  - Manages beams for single request         │
│  - Active beams: list[BeamSearchSequence]   │
│  - Completed: list[BeamSearchSequence]      │
└────────────┬────────────────────────────────┘
             │
             │ Contains multiple
             │
             ▼
┌─────────────────────────────────────────────┐
│       BeamSearchSequence                    │
│  - tokens: list[int]                        │
│  - logprobs: list[dict[int, Logprob]]       │
│  - cum_logprob: float                       │
│  - text: str (optional)                     │
└─────────────────────────────────────────────┘
```

### Beam Search Flow

```
Initial State: Single beam with prompt tokens
    ↓
For each generation step:
    1. Expand each beam → K new candidates
    2. Score all candidates
    3. Select top-N beams
    4. Mark EOS beams as completed
    ↓
Final State: Ranked completed sequences
```

## Data Structures

### BeamSearchSequence

```python
@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    # The tokens include the prompt.
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    multi_modal_data: Optional["MultiModalDataDict"] = None
    mm_processor_kwargs: dict[str, Any] | None = None
```

**Fields:**

- `tokens`: Complete token sequence including prompt
- `logprobs`: Per-token log probabilities with alternatives
- `lora_request`: Optional LoRA adapter configuration
- `cum_logprob`: Cumulative log probability (sum of token logprobs)
- `text`: Decoded text string (populated at output time)
- `finish_reason`: Why sequence ended (e.g., "length", "stop")
- `stop_reason`: Specific stop token or string
- `multi_modal_data`: Multi-modal inputs (images, etc.)
- `mm_processor_kwargs`: Multi-modal processor arguments

**Key Characteristics:**
- **Immutable tokens**: Tokens list includes entire history
- **Incremental logprobs**: New entries appended during generation
- **Lazy text decoding**: Text only computed when needed

### BeamSearchOutput

```python
@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """

    sequences: list[BeamSearchSequence]
```

**Structure:**
- Simple wrapper around list of sequences
- Returned to user after beam search completes
- Sequences sorted by score (best first)

**Usage Pattern:**
```python
output = BeamSearchOutput(sequences=sorted_sequences)
for seq in output.sequences:
    print(f"Score: {seq.cum_logprob:.4f}")
    print(f"Text: {seq.text}")
```

### BeamSearchInstance

```python
class BeamSearchInstance:
    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []
```

**State Management:**

- `beams`: Active beams being expanded
- `completed`: Beams that reached EOS or max length

**Initialization:**
- Starts with single beam containing prompt tokens
- Empty completed list
- Optional logprobs from prompt processing

**Lifecycle:**
```
Init: beams=[prompt], completed=[]
    ↓
Step 1: beams=[seq1, seq2, seq3], completed=[]
    ↓
Step 2: beams=[seq1, seq3, seq4], completed=[seq2]
    ↓
Final: beams=[], completed=[seq1, seq2, seq3, seq4]
```

## Scoring Functions

### get_beam_search_score

```python
def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from
    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)
```

**Algorithm:**

1. **Get sequence length**: Count all tokens
2. **Exclude EOS if present**: Don't penalize EOS token
3. **Apply length penalty**: Divide by `length^penalty`
4. **Return normalized score**: Higher is better

**Length Penalty Effects:**

- `penalty = 0.0`: No normalization (favors longer sequences)
- `penalty = 1.0`: Linear normalization (default)
- `penalty > 1.0`: Strong penalty (favors shorter sequences)

**Mathematical Formula:**
```
score = cumulative_logprob / (seq_len ^ length_penalty)

where:
  cumulative_logprob = sum(log P(token_i | context))
  seq_len = number of generated tokens (excluding EOS)
```

**Example:**
```python
# Sequence 1: 5 tokens, logprob = -10
score1 = -10 / (5 ** 1.0) = -2.0

# Sequence 2: 10 tokens, logprob = -15
score2 = -15 / (10 ** 1.0) = -1.5

# Sequence 2 has higher score (better) despite lower absolute logprob
```

### create_sort_beams_key_function

```python
def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_id, length_penalty
        )

    return sort_beams_key
```

**Purpose:** Creates a key function for sorting beams by score

**Usage Pattern:**
```python
# Create sorting key
key_fn = create_sort_beams_key_function(
    eos_token_id=tokenizer.eos_token_id,
    length_penalty=1.0
)

# Sort beams (highest score first)
sorted_beams = sorted(beams, key=key_fn, reverse=True)

# Select top-k beams
top_k_beams = sorted_beams[:beam_width]
```

**Why Factory Pattern?**
- Avoids passing `eos_token_id` and `length_penalty` repeatedly
- Creates closure with fixed parameters
- More efficient for repeated sorting

## Beam Search Algorithm (Usage)

### Initialization

```python
# Create beam search instance
instance = BeamSearchInstance(
    prompt_tokens=[101, 2023, 2003],  # "This is"
    lora_request=None,
    logprobs=None
)

beam_width = 3
length_penalty = 1.0
eos_token_id = 102
```

### Generation Loop (Conceptual)

```python
while instance.beams and generation_not_complete:
    new_candidates = []

    # Expand each beam
    for beam in instance.beams:
        # Get top-k next tokens from model
        next_token_logprobs = model.get_next_token_logprobs(beam.tokens)
        top_k_tokens = next_token_logprobs.topk(k=beam_width)

        for token_id, logprob in top_k_tokens:
            # Create new candidate
            new_beam = BeamSearchSequence(
                tokens=beam.tokens + [token_id],
                logprobs=beam.logprobs + [next_token_logprobs],
                cum_logprob=beam.cum_logprob + logprob,
                lora_request=beam.lora_request,
            )

            # Check if complete
            if token_id == eos_token_id or len(new_beam.tokens) >= max_len:
                instance.completed.append(new_beam)
            else:
                new_candidates.append(new_beam)

    # Select top beams for next iteration
    key_fn = create_sort_beams_key_function(eos_token_id, length_penalty)
    instance.beams = sorted(new_candidates, key=key_fn, reverse=True)[:beam_width]

# Return completed sequences
key_fn = create_sort_beams_key_function(eos_token_id, length_penalty)
final_sequences = sorted(instance.completed, key=key_fn, reverse=True)
output = BeamSearchOutput(sequences=final_sequences[:beam_width])
```

### Example Trace

```
Step 0:
  Active: [["This", "is"]]
  Completed: []

Step 1:
  Expand beam:
    - "This is a" (score: -2.5)
    - "This is the" (score: -2.3)
    - "This is my" (score: -2.8)
  Select top-3:
    Active: [["This", "is", "the"], ["This", "is", "a"], ["This", "is", "my"]]

Step 2:
  Expand beams:
    From "This is the":
      - "This is the best" (score: -3.1)
      - "This is the only" (score: -3.3)
    From "This is a":
      - "This is a test" (score: -3.2)
      - "This is a good" (score: -3.0)
    From "This is my":
      - "This is my first" (score: -3.5)

  Select top-3:
    Active: [
      ["This", "is", "a", "good"],
      ["This", "is", "the", "best"],
      ["This", "is", "a", "test"]
    ]

... continue until EOS or max_len ...

Final:
  Completed (sorted): [
    "This is a good example." (score: -5.2),
    "This is the best option." (score: -5.5),
    "This is my first try." (score: -6.1)
  ]
```

## Multi-Modal Support

### Multi-Modal Sequence

```python
beam = BeamSearchSequence(
    tokens=[101, 2023, ...],
    logprobs=[...],
    multi_modal_data={"image": image_tensor},
    mm_processor_kwargs={"image_sizes": [(224, 224)]}
)
```

**Use Case:** Vision-language models generating captions or answers

**Propagation:** Multi-modal data carried through all beam expansions

## Performance Characteristics

### Time Complexity

**Per Generation Step:**
- Expand beams: `O(beam_width × vocab_size)` model forward passes
- Sort candidates: `O(n log n)` where `n = beam_width²`
- Select top-k: `O(n)`

**Total:** `O(max_len × beam_width × vocab_size)`

### Memory Complexity

**Active beams:** `O(beam_width × max_len × hidden_dim)`
**Completed beams:** `O(completed_count × max_len)`
**Logprobs history:** `O(beam_width × max_len × top_k)`

### Comparison with Other Decoding Methods

| Method | Quality | Speed | Memory | Diversity |
|--------|---------|-------|--------|-----------|
| Greedy | Low | Fast | Low | None |
| Sampling | Medium | Fast | Low | High |
| Beam Search | High | Slow | Medium | Low-Medium |
| Diverse Beam | High | Slow | Medium | High |

## Integration with vLLM

### Scheduler Integration

The beam search structures are used by vLLM's scheduler to:
1. Track multiple sequences per request
2. Manage KV cache for each beam
3. Schedule beam expansions efficiently
4. Handle beam completion and pruning

### Output Processing

```python
def finalize_beam_output(instance: BeamSearchInstance,
                         tokenizer) -> BeamSearchOutput:
    """Convert beam search instance to output."""
    # Sort by score
    key_fn = create_sort_beams_key_function(eos_token_id, length_penalty)
    sorted_beams = sorted(instance.completed, key=key_fn, reverse=True)

    # Decode text
    for beam in sorted_beams:
        beam.text = tokenizer.decode(beam.tokens)
        beam.finish_reason = "stop" if beam.tokens[-1] == eos_token_id else "length"

    return BeamSearchOutput(sequences=sorted_beams)
```

## Use Cases

### 1. Machine Translation

```python
# Translate with beam search for better quality
prompt = "Translate to French: Hello, how are you?"
beam_width = 5
length_penalty = 1.0
```

**Benefits:** Explores translation alternatives, finds fluent outputs

### 2. Summarization

```python
# Generate summary with multiple candidates
prompt = "Summarize: [long document]"
beam_width = 4
length_penalty = 0.8  # Prefer slightly longer summaries
```

**Benefits:** Balances brevity and completeness

### 3. Code Generation

```python
# Generate code with exploration
prompt = "def fibonacci(n):"
beam_width = 3
length_penalty = 1.0
```

**Benefits:** Finds syntactically correct implementations

### 4. Constrained Generation

```python
# Generate with format constraints
# (would require additional constraint logic)
prompt = "Generate JSON: {name, age, city}"
beam_width = 5
```

**Benefits:** Can integrate with constraint satisfaction

## Advanced Considerations

### Diverse Beam Search

**Extension:** Group beams to encourage diversity
```python
# Conceptual extension (not in current implementation)
class DiverseBeamSearchInstance:
    def __init__(self, num_groups, beams_per_group):
        self.groups = [
            BeamSearchInstance(...) for _ in range(num_groups)
        ]
        self.diversity_penalty = 0.5
```

### Constrained Beam Search

**Extension:** Add constraint satisfaction
```python
# Conceptual extension
def is_valid_candidate(beam: BeamSearchSequence,
                       constraints: List[Constraint]) -> bool:
    return all(c.check(beam.tokens) for c in constraints)
```

### N-Best Hypotheses

```python
# Return multiple best sequences
output = BeamSearchOutput(sequences=top_n_beams)
for i, seq in enumerate(output.sequences):
    print(f"Hypothesis {i+1}: {seq.text} (score: {seq.cum_logprob})")
```

## Best Practices

1. **Choose Beam Width**: Balance quality vs. speed (typical: 4-10)
2. **Tune Length Penalty**: Adjust based on task (translation: ~0.6-1.0)
3. **Set Max Length**: Prevent runaway generation
4. **Monitor Memory**: Beam search uses more memory than greedy
5. **Compare with Sampling**: Not always better than good sampling
6. **Use for Quality-Critical Tasks**: Translation, summarization, QA

## Dependencies

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict
```

**Key Imports:**
- `Logprob`: Log probability data structure
- `LoRARequest`: LoRA adapter configuration
- `MultiModalDataDict`: Multi-modal data handling

## Related Components

- **Sampling Algorithms**: Alternative decoding strategies
- **Scheduler**: Manages beam search execution
- **KV Cache Manager**: Handles cache for multiple beams
- **Output Processor**: Converts sequences to final output
- **Tokenizer**: Decodes token sequences to text

## References

- **Source File**: `vllm/beam_search.py`
- **Transformers Implementation**: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py
- **Original Paper**: "Beam Search" (Reddy, 1977)
- **Length Penalty**: Google's Neural Machine Translation System (Wu et al., 2016)
- **Repository**: https://github.com/vllm-project/vllm

## Summary

The `beam_search.py` module provides essential data structures and algorithms for beam search decoding in vLLM. Through `BeamSearchSequence`, `BeamSearchInstance`, and `BeamSearchOutput`, it enables exploration of multiple generation hypotheses with length-normalized scoring. The implementation supports advanced features like LoRA adapters and multi-modal inputs, making it suitable for high-quality generation tasks where greedy decoding or sampling may be insufficient. While slower than alternatives, beam search remains valuable for quality-critical applications like translation and summarization.

---

*This implementation is a core component of vLLM's text generation capabilities, providing an alternative decoding strategy for improved output quality.*
