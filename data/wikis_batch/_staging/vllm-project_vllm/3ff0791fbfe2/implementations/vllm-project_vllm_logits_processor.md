# Implementation: Logits Processor for Bad Words

**File:** `/tmp/praxium_repo_583nq7ea/vllm/logits_process.py` (121 lines)
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The `logits_process.py` module implements logits processing functionality to prevent the generation of prohibited words. It provides the `NoBadWordsLogitsProcessor` class and a factory function to create processors that can filter out specific words during text generation by applying negative bias to their token probabilities.

**Key Components:**
- `LogitsProcessor` type alias for processor functions
- `NoBadWordsLogitsProcessor` class for bad word filtering
- `get_bad_words_logits_processors()` factory function
- Support for both single-token and multi-token bad words

## Implementation Details

### Type Definition: LogitsProcessor

```python
LogitsProcessor: TypeAlias = (
    Callable[[list[int], torch.Tensor], torch.Tensor]
    | Callable[[list[int], list[int], torch.Tensor], torch.Tensor]
)
"""LogitsProcessor is a function that takes a list
of previously generated tokens, the logits tensor
for the next token and, optionally, prompt tokens as a
first argument, and returns a modified tensor of logits
to sample from."""
```

**Design:**
- **Type Alias Pattern**: Defines standard interface for logits processors
- **Flexible Signature**: Supports both 2-arg and 3-arg callables
- **Functional Style**: Processors are functions that transform logits tensors

### Factory Function: get_bad_words_logits_processors()

```python
def get_bad_words_logits_processors(
    bad_words: list[str], tokenizer: TokenizerLike
) -> list[LogitsProcessor]:
    bad_words_ids: list[list[int]] = list()

    for bad_word in bad_words:
        # To prohibit words both at the beginning
        # and in the middle of text
        # (related to add_prefix_space tokenizer parameter)
        for add_prefix_space in [False, True]:
            prefix = " " if add_prefix_space else ""
            prompt = prefix + bad_word.lstrip()

            prompt_token_ids = tokenizer.encode(text=prompt, add_special_tokens=False)

            # If no space at the beginning
            # or if prefix space produces a new word token
            if (not add_prefix_space) or (
                add_prefix_space
                and prompt_token_ids[0] != bad_words_ids[-1][0]
                and len(prompt_token_ids) == len(bad_words_ids[-1])
            ):
                bad_words_ids.append(prompt_token_ids)

    return [NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids)]
```

**Implementation Strategy:**

1. **Tokenization with Variants**: Each bad word is tokenized twice:
   - Without prefix space (e.g., "badword")
   - With prefix space (e.g., " badword")

2. **Rationale**: Handles tokenizer behavior variations:
   - Some tokenizers produce different tokens based on whitespace prefix
   - Ensures bad words are blocked regardless of context position
   - Critical for subword tokenization models (BPE, WordPiece)

3. **Deduplication Logic**: Only adds variant if it produces different tokens:
   ```python
   if (not add_prefix_space) or (
       add_prefix_space
       and prompt_token_ids[0] != bad_words_ids[-1][0]
       and len(prompt_token_ids) == len(bad_words_ids[-1])
   ):
   ```

**Example:**
```python
# Tokenizer example with BPE
tokenizer.encode("badword")        # [1234, 5678]
tokenizer.encode(" badword")       # [999, 1234, 5678]  # Different!

# Both sequences added to bad_words_ids
bad_words_ids = [
    [1234, 5678],        # Without space
    [999, 1234, 5678]    # With space
]
```

### Core Class: NoBadWordsLogitsProcessor

```python
class NoBadWordsLogitsProcessor:
    _SMALLEST_LOGIT = float("-inf")
    _NEUTRAL_LOGIT = 0.0

    def __init__(self, bad_words_ids: list[list[int]]):
        self.bad_words_ids = bad_words_ids
        self.word_bias: torch.FloatTensor = None
```

**Design Choices:**
- **Class Constants**: Use `-inf` for blocking tokens, `0.0` for no effect
- **Lazy Initialization**: `word_bias` tensor created on first call
- **Stateful Design**: Maintains bias tensor across calls for efficiency

### Processing Logic: __call__()

```python
def __call__(
    self,
    past_tokens_ids: Sequence[int],
    logits: torch.FloatTensor,
) -> torch.Tensor:
    if self.word_bias is None:
        self._init_word_bias(logits=logits)

    last_token_bias = torch.zeros_like(logits)

    for bad_word_ids in self.bad_words_ids:
        if len(bad_word_ids) == 1:  # 1-token words already processed
            continue

        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        actual_prefix = past_tokens_ids[-prefix_length:]
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        is_match = tuple(actual_prefix) == tuple(expected_prefix)
        last_token_bias[last_token_id] += (
            self._SMALLEST_LOGIT if is_match else self._NEUTRAL_LOGIT
        )

    logits = logits + self.word_bias + last_token_bias

    return logits
```

**Algorithm Breakdown:**

#### 1. Initialization Phase
```python
if self.word_bias is None:
    self._init_word_bias(logits=logits)
```
- Creates static bias tensor for single-token bad words
- Only runs once per processor instance

#### 2. Multi-Token Word Processing

For each bad word sequence:

**Skip Conditions:**
```python
if len(bad_word_ids) == 1:  # Already in word_bias
    continue

if len(bad_word_ids) > len(past_tokens_ids) + 1:  # Prefix too long
    continue
```

**Prefix Matching:**
```python
prefix_length = len(bad_word_ids) - 1
last_token_id = bad_word_ids[-1]
actual_prefix = past_tokens_ids[-prefix_length:]
expected_prefix = bad_word_ids[:prefix_length]

is_match = tuple(actual_prefix) == tuple(expected_prefix)
```

**Example:**
```python
# Bad word: [123, 456, 789] (3 tokens)
# Past tokens: [..., 999, 123, 456]

prefix_length = 2
expected_prefix = [123, 456]
actual_prefix = [123, 456]  # Last 2 tokens
is_match = True  # Match! Block token 789
```

#### 3. Bias Application

```python
last_token_bias[last_token_id] += (
    self._SMALLEST_LOGIT if is_match else self._NEUTRAL_LOGIT
)

logits = logits + self.word_bias + last_token_bias
```

**Bias Types:**
- `word_bias`: Static bias for single-token bad words (always applied)
- `last_token_bias`: Dynamic bias for multi-token bad words (context-dependent)

### Initialization: _init_word_bias()

```python
def _init_word_bias(self, logits: torch.FloatTensor) -> None:
    vocab_size = logits.shape[-1]

    self._check_token_ids_bounds(vocab_size=vocab_size)

    self.word_bias = torch.zeros(
        (vocab_size,), dtype=torch.float, device=logits.device
    )

    for bad_word_ids in self.bad_words_ids:
        if len(bad_word_ids) == 1:
            bad_word_id = bad_word_ids[-1]
            self.word_bias[bad_word_id] = self._SMALLEST_LOGIT
```

**Implementation Details:**
- Creates zero-initialized bias tensor matching logits device/vocab size
- Sets `-inf` bias for all single-token bad words
- Called lazily to infer device and vocabulary size from first logits tensor

### Validation: _check_token_ids_bounds()

```python
def _check_token_ids_bounds(self, vocab_size: int) -> None:
    invalid_token_ids = []

    for bad_word_ids in self.bad_words_ids:
        for token_id in bad_word_ids:
            if token_id < 0 or token_id >= vocab_size:
                invalid_token_ids.append(token_id)

    if len(invalid_token_ids) > 0:
        raise ValueError(
            f"The model vocabulary size is {vocab_size},"
            f" but the following tokens"
            f" were specified as bad: {invalid_token_ids}."
            f" All token id values should be integers satisfying:"
            f" 0 <= token_id < {vocab_size}."
        )
```

**Validation Strategy:**
- Checks all token IDs are within vocabulary bounds
- Provides detailed error message with invalid tokens
- Prevents out-of-bounds tensor indexing

## Usage Patterns

### Basic Usage

```python
from vllm.logits_process import get_bad_words_logits_processors

# Create processor for bad words
bad_words = ["profanity", "offensive", "inappropriate"]
processors = get_bad_words_logits_processors(bad_words, tokenizer)

# Apply during generation
for processor in processors:
    logits = processor(past_token_ids, logits)
```

### Integration with Sampling

```python
# In sampling loop
while not done:
    # Get logits from model
    logits = model(input_ids)

    # Apply bad words filter
    for processor in logits_processors:
        logits = processor(past_tokens, logits)

    # Sample next token
    next_token = sample(logits)
    past_tokens.append(next_token)
```

### Custom LogitsProcessor

```python
def custom_processor(past_tokens: list[int], logits: torch.Tensor) -> torch.Tensor:
    # Custom logic here
    return modified_logits

# Use alongside bad words processor
all_processors = [
    *get_bad_words_logits_processors(bad_words, tokenizer),
    custom_processor
]
```

## Algorithm Complexity

### Time Complexity

**Per Call:**
- Single-token filtering: O(1) - just tensor addition
- Multi-token filtering: O(B × L) where:
  - B = number of bad word sequences
  - L = average length of bad word sequences

**Prefix Matching:**
- O(L) comparison per bad word sequence
- Total: O(B × L) per generation step

### Space Complexity

- `word_bias`: O(V) where V is vocabulary size
- `last_token_bias`: O(V) per call
- `bad_words_ids`: O(B × L) stored permanently

## Performance Considerations

### Optimization Strategies

1. **Static Bias Precomputation**: Single-token bad words processed once
2. **Early Exit**: Skips checks for impossible matches
3. **Batch Processing**: Operates on full logits tensor, not per-token

### Memory Efficiency

```python
# word_bias created once per processor
self.word_bias = torch.zeros((vocab_size,), dtype=torch.float, device=logits.device)

# last_token_bias created per call, but quickly garbage collected
last_token_bias = torch.zeros_like(logits)
```

### Device Placement

The bias tensors are automatically placed on the same device as the logits:
```python
device=logits.device  # Matches input device (CPU/GPU)
```

## Integration Points

### Sampling Parameters

Used in conjunction with `SamplingParams`:
```python
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    logits_processors=get_bad_words_logits_processors(bad_words, tokenizer)
)
```

### Generation Pipeline

Integrated into the generation loop:
1. Model forward pass produces logits
2. Logits processors modify logits
3. Sampling draws from modified distribution
4. Process repeats for next token

### OpenAI API Compatibility

Supports OpenAI's `logit_bias` parameter:
```python
# Can be combined with manual logit biases
all_processors = [
    *get_bad_words_logits_processors(bad_words, tokenizer),
    manual_logit_bias_processor
]
```

## Comparison to HuggingFace Transformers

This implementation is based on HuggingFace's processors:
```python
# Code based on NoBadWordsLogitsProcessor and SequenceBiasLogitsProcessor
# from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
```

**Key Differences:**
- Simplified for vLLM's use case
- Optimized for batch processing
- Integrated with vLLM's sampling pipeline

## Edge Cases

### Empty Bad Words List

```python
processors = get_bad_words_logits_processors([], tokenizer)
# Returns processor that does nothing (word_bias all zeros)
```

### Bad Word at Generation Start

```python
past_tokens = []  # Empty history
# Multi-token bad words with len > 1 are skipped (prefix too long)
# Only single-token bad words are blocked
```

### Overlapping Bad Word Sequences

```python
bad_words = ["bad", "badword"]
# Both tokenized: [123] and [123, 456]
# Token 123 always blocked (single-token)
# Token 456 blocked only after 123 (multi-token)
```

### Invalid Token IDs

```python
# If tokenizer produces out-of-bounds IDs
try:
    processor = NoBadWordsLogitsProcessor([[99999]])
    processor([], logits)  # vocab_size = 50000
except ValueError as e:
    # "The model vocabulary size is 50000, but the following tokens
    #  were specified as bad: [99999]..."
```

## Design Rationale

### Why Process Both With/Without Prefix Space?

Different tokenizers handle whitespace differently:
```python
# GPT-2 style tokenizer
tokenize("badword")   # ['bad', 'word']
tokenize(" badword")  # [' bad', 'word']  # Different first token!

# Need to block both variants to catch all occurrences
```

### Why Use -inf Instead of 0?

```python
self._SMALLEST_LOGIT = float("-inf")
```

- Ensures zero probability after softmax: `softmax(-inf) = 0`
- More reliable than large negative numbers
- Explicitly indicates "impossible" rather than "unlikely"

### Why Lazy Initialization?

```python
if self.word_bias is None:
    self._init_word_bias(logits=logits)
```

Benefits:
1. **Device Inference**: Learns device from first logits tensor
2. **Vocabulary Inference**: Learns vocab size from logits shape
3. **Memory Efficiency**: No allocation until first use

## Testing Considerations

### Test Cases

1. **Single-token bad words**: Verify blocked at all positions
2. **Multi-token bad words**: Verify prefix matching works
3. **Prefix space variants**: Test both tokenization variants
4. **Edge cases**: Empty list, invalid tokens, generation start
5. **Device compatibility**: Test CPU and GPU placement

### Mock Example

```python
def test_bad_words():
    # Mock tokenizer
    tokenizer = MockTokenizer()

    # Create processor
    processors = get_bad_words_logits_processors(["bad"], tokenizer)

    # Mock logits
    logits = torch.zeros(vocab_size)
    logits[bad_token_id] = 10.0  # High logit for bad token

    # Apply processor
    filtered_logits = processors[0]([], logits)

    # Verify bad token has -inf logit
    assert filtered_logits[bad_token_id] == float("-inf")
```

## Related Components

- **vllm.sampling_params**: Integrates logits processors into sampling
- **vllm.tokenizers**: Provides tokenizer interface for word encoding
- **vllm.model_executor.layers.sampler**: Applies processors during sampling
- **HuggingFace Transformers**: Original inspiration for implementation

## Future Enhancements

1. **Good Words Boost**: Positive bias for encouraged words
2. **Pattern Matching**: Regex-based token filtering
3. **Context-Aware Filtering**: Different bad words for different contexts
4. **Performance Optimization**: Caching prefix matches across batches
5. **Configurable Bias Values**: Allow values other than -inf
6. **Phrase Blocking**: Block specific multi-token phrases more efficiently

## Security Considerations

### Content Filtering

Used for:
- Preventing generation of profanity
- Blocking sensitive terms (SSNs, credit cards)
- Enforcing content policies

**Limitations:**
- Only works for exact token matches
- Subword tokenization may split bad words unexpectedly
- Determined users might find workarounds (homoglyphs, spaces)

### Prompt Injection Defense

Can help mitigate prompt injection by blocking:
- System prompt leakage attempts
- Instruction override keywords
- Sensitive internal terms

## Summary

The `logits_process.py` module provides efficient bad word filtering through logits modification. Its dual-path approach (static bias for single tokens, dynamic matching for multi-token sequences) balances performance with effectiveness. The tokenization variant handling ensures robustness across different tokenizer types, making it a reliable component for content filtering and constrained generation in vLLM.
