# Implementation: Log Probability Data Structures

**File:** `/tmp/praxium_repo_583nq7ea/vllm/logprobs.py` (206 lines)
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The `logprobs.py` module defines data structures for storing and managing log probabilities during text generation. It provides two main representations: the simple `Logprob` dataclass for individual token information, and the optimized `FlatLogprobs` container that significantly reduces garbage collection overhead by flattening nested dictionaries into primitive-type lists.

**Key Components:**
- `Logprob`: Individual token log probability with rank and decoded text
- `FlatLogprobs`: Memory-efficient container implementing `MutableSequence`
- Helper functions for creating and populating logprob containers
- Type aliases for prompt and sample logprobs

## Implementation Details

### Core Data Structure: Logprob

```python
@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None
```

**Design Choices:**
- **Dataclass Pattern**: Simple, immutable-like structure for single token info
- **Optional Fields**: `rank` and `decoded_token` can be None when not requested
- **OpenAI Compatibility**: Matches OpenAI's logprobs response format

**Usage:**
```python
# Token with full information
token_info = Logprob(
    logprob=-0.5,
    rank=3,
    decoded_token="hello"
)

# Minimal token info (rank/decoded optional)
token_info = Logprob(logprob=-1.2)
```

### Type Aliases

```python
LogprobsOnePosition = dict[int, Logprob]
```

Represents logprobs for all considered tokens at a single position:
```python
# Example: top-3 tokens at position 5
position_logprobs: LogprobsOnePosition = {
    1234: Logprob(logprob=-0.1, rank=1, decoded_token="the"),
    5678: Logprob(logprob=-0.5, rank=2, decoded_token="a"),
    9012: Logprob(logprob=-1.0, rank=3, decoded_token="an")
}
```

### Optimized Container: FlatLogprobs

```python
@dataclass
class FlatLogprobs(MutableSequence[LogprobsOnePosition]):
    """
    Flat logprobs of a request into multiple primitive type lists.

    Compared to list[dict[int, Logprob]], this data structure reduced GC
    overhead significantly. As it flattened logprob information for
    all positions and ranks in to multiple primitive type lists (i.e.
    logprobs, token_ids, ranks per token_ids, decoded_tokens).
    So regardless of the sequence length and top_logprobs setup,
    FlatLogprobs would only introduce a constant amount of objects.
    """

    # Start / end indices to indicate the range of logprobs for each position.
    start_indices: list[int] = field(default_factory=list)
    end_indices: list[int] = field(default_factory=list)

    # Flatten Logprob information for (each position, rank).
    # For position <i>, the logprobs are ranged
    # from self.start_indices[i] to self.end_indices[i] (exclusive).
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    ranks: list[int | None] = field(default_factory=list)
    decoded_tokens: list[str | None] = field(default_factory=list)
```

**Optimization Rationale:**

Traditional approach (high GC overhead):
```python
# Creates many objects: list, dicts, Logprob instances
traditional: list[dict[int, Logprob]] = [
    {1: Logprob(-0.1, 1, "a"), 2: Logprob(-0.5, 2, "b")},  # Position 0
    {3: Logprob(-0.2, 1, "c"), 4: Logprob(-0.6, 2, "d")},  # Position 1
    # ... hundreds more positions
]
# Object count: 1 list + N dicts + N*K Logprobs = O(N*K) objects
```

FlatLogprobs approach (low GC overhead):
```python
# Only 6 list objects total, regardless of sequence length!
flat = FlatLogprobs(
    start_indices=[0, 2],           # 1 list
    end_indices=[2, 4],              # 1 list
    token_ids=[1, 2, 3, 4],          # 1 list
    logprobs=[-0.1, -0.5, -0.2, -0.6], # 1 list
    ranks=[1, 2, 1, 2],              # 1 list
    decoded_tokens=["a", "b", "c", "d"] # 1 list
)
# Object count: 6 lists (constant)
```

### FlatLogprobs: Index Management

```python
# Start / end indices to indicate the range of logprobs for each position.
start_indices: list[int] = field(default_factory=list)
end_indices: list[int] = field(default_factory=list)
```

**Index Scheme:**
- `start_indices[i]`: Start of logprobs for position i
- `end_indices[i]`: End of logprobs for position i (exclusive)
- Data for position i: `token_ids[start:end]`, `logprobs[start:end]`, etc.

**Example:**
```python
flat = FlatLogprobs(
    start_indices=[0, 3, 5],  # 3 positions
    end_indices=[3, 5, 8],
    token_ids=[10, 20, 30, 40, 50, 60, 70, 80],
    logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
    # ...
)

# Position 0: indices [0:3] -> tokens [10, 20, 30]
# Position 1: indices [3:5] -> tokens [40, 50]
# Position 2: indices [5:8] -> tokens [60, 70, 80]
```

### FlatLogprobs: Append Methods

#### Standard Append

```python
def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None:
    """Appends the container with logprobs for the next position"""
    self.start_indices.append(len(self.logprobs))
    if logprobs_one_position:
        for token_id, logprob in logprobs_one_position.items():
            self.token_ids.append(token_id)
            self.logprobs.append(logprob.logprob)
            self.ranks.append(logprob.rank)
            self.decoded_tokens.append(logprob.decoded_token)
    self.end_indices.append(len(self.logprobs))
```

**Implementation:**
1. Record current length as start index
2. Append all token data for this position
3. Record new length as end index

**Usage:**
```python
flat = FlatLogprobs()
flat.append({
    100: Logprob(-0.1, 1, "hello"),
    200: Logprob(-0.5, 2, "hi")
})
# Result: start_indices=[0], end_indices=[2], token_ids=[100, 200], ...
```

#### Fast Append (Optimization)

```python
def append_fast(
    self,
    token_ids: list[int],
    logprobs: list[float],
    ranks: itertools.chain[int],
    decoded_tokens: Iterable[str | None],
) -> None:
    """
    Appends logprobs for the next position without creating
    the intermediate logprob dictionary.
    """
    self.start_indices.append(len(self.logprobs))
    for token_id, logprob, rank, decoded_token in zip(
        token_ids, logprobs, ranks, decoded_tokens
    ):
        self.token_ids.append(token_id)
        self.logprobs.append(logprob)
        self.ranks.append(rank)
        self.decoded_tokens.append(decoded_token)
    self.end_indices.append(len(self.logprobs))
```

**Performance Benefit:**
- Skips creation of intermediate `dict[int, Logprob]`
- Directly appends primitive values
- Used in hot path for maximum performance

**Usage:**
```python
# In performance-critical code
flat.append_fast(
    token_ids=[100, 200],
    logprobs=[-0.1, -0.5],
    ranks=itertools.chain([1], range(1, 3)),
    decoded_tokens=["hello", "hi"]
)
```

### FlatLogprobs: MutableSequence Interface

#### Length

```python
def __len__(self) -> int:
    """Gets number of positions stored in the container"""
    return len(self.start_indices)
```

Number of positions, not total number of logprobs.

#### Getitem (Indexing)

```python
@overload
def __getitem__(self, position: int) -> LogprobsOnePosition: ...

@overload
def __getitem__(self, s: slice, /) -> "FlatLogprobs": ...

def __getitem__(self, index: int | slice):
    """Extracts logprobs of a given position or slice"""
    if isinstance(index, int):
        return {
            self.token_ids[i]: Logprob(
                logprob=self.logprobs[i],
                rank=self.ranks[i],
                decoded_token=self.decoded_tokens[i],
            )
            for i in range(self.start_indices[index], self.end_indices[index])
        }
    elif isinstance(index, slice):
        min_index = self.start_indices[index][0]
        max_index = self.end_indices[index][-1]
        return FlatLogprobs(
            # Shift updated start_indices and end_indices to
            # be 0-indexed
            start_indices=[i - min_index for i in self.start_indices[index]],
            end_indices=[i - min_index for i in self.end_indices[index]],
            token_ids=self.token_ids[min_index:max_index],
            logprobs=self.logprobs[min_index:max_index],
            ranks=self.ranks[min_index:max_index],
            decoded_tokens=self.decoded_tokens[min_index:max_index],
        )
    else:
        raise TypeError(f"Invalid index type: {type(index)}")
```

**Integer Indexing:**
```python
flat = FlatLogprobs(...)
position_5_logprobs: LogprobsOnePosition = flat[5]
# Returns: {token_id: Logprob(...), ...}
```

**Slice Indexing:**
```python
flat = FlatLogprobs(...)
subsequence: FlatLogprobs = flat[10:20]
# Returns new FlatLogprobs with re-indexed data
```

**Slice Implementation Details:**
- Extracts relevant portion of each flat list
- Re-indexes start/end indices to be 0-based
- Preserves FlatLogprobs structure

#### Iterator

```python
def __iter__(self) -> Iterator[LogprobsOnePosition]:
    """
    Iterates the container and yields LogprobsOnePosition for
    each position.
    """
    for i in range(0, len(self.start_indices)):
        yield self.__getitem__(i)
```

**Usage:**
```python
for position_logprobs in flat:
    # position_logprobs is dict[int, Logprob]
    for token_id, logprob in position_logprobs.items():
        print(f"Token {token_id}: {logprob.logprob}")
```

#### Immutable Operations

```python
def __setitem__(self, item, value) -> None:
    raise TypeError("Cannot set logprobs in FlatLogprobs")

def __delitem__(self, item) -> None:
    raise TypeError("Cannot delete logprobs from FlatLogprobs")

def insert(self, item) -> None:
    raise TypeError("Cannot insert logprobs to FlatLogprobs")
```

**Design Decision:**
- FlatLogprobs is append-only
- Prevents accidental mutation that could corrupt index structure
- Simplifies reasoning about data consistency

### Type Aliases for Usage Context

```python
# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]

# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]
```

**Purpose:**
- Documents expected usage contexts
- Distinguishes prompt vs. sample logprobs
- Allows both flat and traditional representations

### Factory Functions

#### Create Prompt Logprobs

```python
def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs:
    """Creates a container to store prompt logprobs for a request"""
    logprobs = FlatLogprobs() if flat_logprobs else []
    # NOTE: logprob of first prompt token is None.
    logprobs.append(None)
    return logprobs
```

**Initial State:**
- First position is None (no logprob for initial token)
- Consistent with OpenAI API behavior

#### Create Sample Logprobs

```python
def create_sample_logprobs(flat_logprobs: bool) -> SampleLogprobs:
    """Creates a container to store decode logprobs for a request"""
    return FlatLogprobs() if flat_logprobs else []
```

**Difference from Prompt:**
- No initial None entry
- All generated tokens have logprobs

### Helper Function: Append Logprobs

```python
def append_logprobs_for_next_position(
    request_logprobs: PromptLogprobs | SampleLogprobs,
    token_ids: list[int],
    logprobs: list[float],
    decoded_tokens: Iterable[str | None],
    rank: int,
    num_logprobs: int,
) -> None:
    """Appends logprobs for the next position"""
    if num_logprobs == -1:
        num_logprobs = len(logprobs)
    # We do not need a special case for the sampled token
    # being in the topk, since inserting duplicated data
    # into a dictionary twice is the same as doing it once.
    topk_ranks = range(1, num_logprobs + 1)
    ranks = itertools.chain((rank,), topk_ranks)

    if isinstance(request_logprobs, FlatLogprobs):
        request_logprobs.append_fast(token_ids, logprobs, ranks, decoded_tokens)
    else:
        request_logprobs.append(
            {
                token_id: Logprob(
                    logprob=logprob,
                    rank=rank,
                    decoded_token=token,
                )
                for token_id, logprob, rank, token in zip(
                    token_ids, logprobs, ranks, decoded_tokens
                )
            }
        )
```

**Implementation Details:**

1. **Rank Generation:**
```python
topk_ranks = range(1, num_logprobs + 1)  # [1, 2, 3, ...]
ranks = itertools.chain((rank,), topk_ranks)  # [sampled_rank, 1, 2, 3, ...]
```

2. **Duplicate Handling:**
- Sampled token might also be in top-k
- Dictionary insert handles duplicates naturally (last write wins)
- No special case needed

3. **Type Dispatch:**
- Uses `append_fast()` for FlatLogprobs (performance)
- Uses standard dict construction for list-based storage

## Usage Patterns

### Basic Usage

```python
# Create container
flat = create_sample_logprobs(flat_logprobs=True)

# Append logprobs for each generated token
for token_id, logprob_value, token_text in generation:
    append_logprobs_for_next_position(
        request_logprobs=flat,
        token_ids=[token_id],
        logprobs=[logprob_value],
        decoded_tokens=[token_text],
        rank=1,
        num_logprobs=1
    )

# Access logprobs
for position_idx, position_logprobs in enumerate(flat):
    print(f"Position {position_idx}: {position_logprobs}")
```

### With Top-K Logprobs

```python
# Request top-5 logprobs
flat = create_sample_logprobs(flat_logprobs=True)

# After sampling
sampled_token = 1234
topk_tokens = [1234, 5678, 9012, 3456, 7890]
topk_logprobs = [-0.1, -0.5, -1.0, -1.5, -2.0]
topk_texts = ["the", "a", "an", "to", "of"]

append_logprobs_for_next_position(
    request_logprobs=flat,
    token_ids=topk_tokens,
    logprobs=topk_logprobs,
    decoded_tokens=topk_texts,
    rank=1,  # sampled token rank
    num_logprobs=5
)
```

### Backward Compatibility

```python
# Old code using list[dict] still works
traditional = create_sample_logprobs(flat_logprobs=False)

# Same interface
append_logprobs_for_next_position(traditional, ...)

# Same iteration pattern
for position_logprobs in traditional:
    # Same dict[int, Logprob] structure
    pass
```

### Slicing for Prefill/Decode Separation

```python
# All logprobs for a request
all_logprobs = FlatLogprobs(...)

# Split at prefill/decode boundary
prefill_logprobs = all_logprobs[:prefill_length]
decode_logprobs = all_logprobs[prefill_length:]
```

## Performance Analysis

### Memory Comparison

**Scenario:** 100 positions, 5 logprobs per position

Traditional approach:
```python
# 1 list + 100 dicts + 500 Logprob objects
memory = sizeof(list) + 100 * sizeof(dict) + 500 * sizeof(Logprob)
# Approximate: 8 bytes + 100 * 240 bytes + 500 * 56 bytes = ~52KB
# Plus GC overhead for tracking 601 objects
```

FlatLogprobs approach:
```python
# 6 lists + 500 primitive values
memory = 6 * sizeof(list) + 500 * sizeof(primitive)
# Approximate: 6 * 56 bytes + 500 * 8 bytes = ~4.3KB
# Plus GC overhead for tracking 6 objects
```

**Savings:** ~90% memory reduction, 99% fewer GC-tracked objects

### GC Impact

Python's garbage collector tracks objects:
- Traditional: 601 objects to track
- FlatLogprobs: 6 objects to track

For long sequences (1000+ tokens) with top-10 logprobs:
- Traditional: 11,000+ objects
- FlatLogprobs: 6 objects (constant)

**Result:** Significantly reduced GC pauses in high-throughput serving.

### Time Complexity

- `append()`: O(K) where K is number of logprobs per position
- `__getitem__(int)`: O(K) to reconstruct dict
- `__getitem__(slice)`: O(N*K) where N is slice size
- `__iter__()`: O(P*K) where P is number of positions

### Space Complexity

- Traditional: O(P * K) objects
- FlatLogprobs: O(1) objects, O(P * K) primitive values

## Integration Points

### OpenAI API Compatibility

```python
# OpenAI API response format
{
    "choices": [{
        "logprobs": {
            "tokens": ["hello", " world"],
            "token_logprobs": [-0.1, -0.5],
            "top_logprobs": [
                {"hello": -0.1, "hi": -0.5, "hey": -1.0},
                {" world": -0.5, " universe": -1.0}
            ]
        }
    }]
}

# Built from FlatLogprobs
for position_idx, position_logprobs in flat_logprobs:
    tokens.append(position_logprobs[sampled_token].decoded_token)
    token_logprobs.append(position_logprobs[sampled_token].logprob)
    top_logprobs.append({
        info.decoded_token: info.logprob
        for token_id, info in position_logprobs.items()
    })
```

### Sampling Pipeline

```python
# In sampler
logprob_dict = {
    token_id: Logprob(logprob=lp, rank=r, decoded_token=text)
    for token_id, lp, r, text in zip(...)
}

# Stored in sequence output
if use_flat:
    seq.output_logprobs.append_fast(...)
else:
    seq.output_logprobs.append(logprob_dict)
```

### Request Output

```python
class RequestOutput:
    prompt_logprobs: PromptLogprobs | None
    outputs: list[CompletionOutput]

class CompletionOutput:
    logprobs: SampleLogprobs | None
```

## Design Rationale

### Why MutableSequence?

Provides familiar list-like interface:
```python
len(flat)           # Number of positions
flat[5]             # Logprobs at position 5
flat[10:20]         # Slice of positions
for pos in flat:    # Iterate positions
```

### Why Append-Only?

1. **Correctness**: Prevents index corruption
2. **Simplicity**: No need to handle mutation edge cases
3. **Performance**: Optimized for sequential generation pattern

### Why Both Flat and Traditional?

1. **Migration Path**: Allows gradual adoption
2. **Debugging**: Traditional format easier to inspect
3. **External APIs**: Some consumers might expect dict format

### Why Separate Start/End Indices?

Alternative: Store lengths instead of end indices
```python
# Alternative design
start_indices: list[int]
lengths: list[int]
```

Chosen design has advantages:
- Faster range computation: `[start:end]` vs. `[start:start+length]`
- More explicit about exclusive end
- Common pattern in Python (similar to `range()`)

## Testing Considerations

### Test Cases

1. **Empty Container:**
```python
flat = FlatLogprobs()
assert len(flat) == 0
assert list(flat) == []
```

2. **Single Position:**
```python
flat = FlatLogprobs()
flat.append({100: Logprob(-0.1, 1, "a")})
assert len(flat) == 1
assert 100 in flat[0]
```

3. **Multiple Positions:**
```python
flat = FlatLogprobs()
for i in range(10):
    flat.append({i: Logprob(-float(i), i, str(i))})
assert len(flat) == 10
```

4. **Slicing:**
```python
subset = flat[2:5]
assert len(subset) == 3
assert subset.start_indices == [0, ...]  # Re-indexed
```

5. **None Handling:**
```python
flat = create_prompt_logprobs(flat_logprobs=True)
assert flat[0] == {}  # First position is None -> empty dict
```

6. **Equivalence:**
```python
# Verify flat and traditional produce same output
flat = create_sample_logprobs(flat_logprobs=True)
trad = create_sample_logprobs(flat_logprobs=False)

for _ in range(10):
    data = generate_test_data()
    append_logprobs_for_next_position(flat, **data)
    append_logprobs_for_next_position(trad, **data)

assert list(flat) == trad
```

## Related Components

- **vllm.sampling_params**: Configures logprobs via `logprobs` parameter
- **vllm.model_executor.layers.sampler**: Generates logprobs during sampling
- **vllm.outputs**: Uses logprobs in RequestOutput and CompletionOutput
- **vllm.entrypoints.openai.protocol**: Converts to OpenAI API format

## Future Enhancements

1. **NumPy Backend**: Use numpy arrays instead of lists for even better performance
2. **Compression**: Compress decoded tokens with shared string pool
3. **Lazy Decoding**: Defer token decoding until access
4. **Memory Mapping**: Support memory-mapped storage for very long sequences
5. **Batching**: Batch multiple sequences in single FlatLogprobs instance

## Summary

The `logprobs.py` module provides a high-performance solution to a critical scalability challenge in LLM serving. By flattening nested data structures into primitive-type lists, `FlatLogprobs` reduces GC overhead by orders of magnitude while maintaining full backward compatibility through its `MutableSequence` interface. This design exemplifies the principle of optimizing hot paths without sacrificing usability or correctness.
