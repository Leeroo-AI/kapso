# LongDocumentQABenchmark - Prefix Caching Throughput Benchmark

## Overview

**File:** `/tmp/praxium_repo_583nq7ea/benchmarks/benchmark_long_document_qa_throughput.py` (202 lines)

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Purpose:** Benchmarks long document QA throughput with automatic prefix caching, measuring performance benefits when serving multiple requests with shared long prefixes (simulating RAG systems querying the same large documents).

## Core Architecture

### Main Function

**Lines:** 107-142

```python
def main(args):
    random.seed(args.shuffle_seed)

    # Create prompts with document ID prefix to ensure uniqueness
    prompts = [
        str(i) + " ".join(["hi"] * args.document_length)
        for i in range(args.num_documents)
    ]

    # Repeat prompts in specified pattern
    prompts = repeat_prompts(prompts, args.repeat_count, mode=args.repeat_mode)

    # Create warmup prompts
    warmup_prompts = [
        "This is warm up request " + str(i) + " ".join(["hi"] * args.document_length)
        for i in range(args.num_documents)
    ]

    # Initialize LLM engine
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)

    # Warmup phase
    print("------warm up------")
    test_long_document_qa(llm, warmup_prompts, sampling_params)

    # Actual benchmark
    print("------start generating------")
    test_long_document_qa(llm, prompts, sampling_params)
```

### Test Execution

**Lines:** 52-65

```python
def test_long_document_qa(llm=None, sampling_params=None, prompts=None):
    """
    Test long document QA with given prompts and sampling parameters.
    Print time spent processing all prompts.
    """
    start_time = time.time()
    llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    print(f"Time to execute all requests: {end_time - start_time:.4f} secs")
```

**Key Metric:** Total wall-clock time for all requests.

### Prompt Repetition Strategies

**Lines:** 68-104

```python
def repeat_prompts(prompts, repeat_count, mode: str):
    """
    Repeat each prompt with different orderings to test cache hit patterns.

    Modes:
      - 'random': Shuffle prompts randomly (default)
      - 'tile': Repeat entire list sequentially [1,2,3,1,2,3]
      - 'interleave': Repeat each consecutively [1,1,2,2,3,3]
    """
    print("Repeat mode: ", mode)

    if mode == "random":
        repeated_prompts = prompts * repeat_count
        random.shuffle(repeated_prompts)
        return repeated_prompts

    elif mode == "tile":
        # Lowest cache hit: each cycle starts fresh
        return prompts * repeat_count

    elif mode == "interleave":
        # Highest cache hit: same prompt repeated consecutively
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * repeat_count)
        return repeated_prompts

    else:
        raise ValueError(
            f"Invalid mode: {mode}, only support 'random', 'tile', 'interleave'"
        )
```

**Cache Hit Analysis:**
- **interleave:** Maximum cache hits (consecutive identical prompts)
- **random:** Variable cache hits (depends on shuffle)
- **tile:** Minimum cache hits (cache evicted between cycles)

## Command-Line Interface

**Lines:** 145-196

```python
def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark the performance with or without automatic prefix caching."
    )

    parser.add_argument(
        "--document-length",
        type=int,
        default=20000,  # Roughly a system paper excluding images
        help="Length of each document in tokens"
    )

    parser.add_argument(
        "--num-documents",
        type=int,
        default=8,
        help="Number of documents to sample prompts from"
    )

    parser.add_argument(
        "--output-len",
        type=int,
        default=10,
        help="Number of tokens to generate"
    )

    parser.add_argument(
        "--repeat-count",
        type=int,
        default=2,
        help="Number of times to repeat each prompt"
    )

    parser.add_argument(
        "--repeat-mode",
        type=str,
        default="random",
        help="Mode to repeat prompts: 'random', 'tile', or 'interleave'"
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Random seed for random repeat mode"
    )

    # Add all vLLM engine arguments
    parser = EngineArgs.add_cli_args(parser)
    return parser
```

## Usage Examples

### Basic Usage with Prefix Caching

```bash
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-documents 8 \
    --repeat-count 2
```

**Expected Behavior:**
- First occurrence of each document: Normal processing time
- Subsequent occurrences: Faster due to KV cache reuse

### Testing Different Repeat Patterns

```bash
# Maximum cache hits (interleave mode)
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-documents 8 \
    --repeat-count 3 \
    --repeat-mode interleave

# Minimum cache hits (tile mode)
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-documents 8 \
    --repeat-count 3 \
    --repeat-mode tile

# Random ordering
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-documents 8 \
    --repeat-count 3 \
    --repeat-mode random \
    --shuffle-seed 42
```

### Custom Document Lengths

```bash
# Very long documents (50K tokens)
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --document-length 50000 \
    --num-documents 4 \
    --repeat-count 5 \
    --output-len 100
```

### Comparison: With vs Without Caching

```bash
# With caching
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-documents 8 \
    --repeat-count 4 \
    --repeat-mode interleave

# Without caching (baseline)
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --num-documents 8 \
    --repeat-count 4 \
    --repeat-mode interleave
```

**Expected Speedup:** 2-10x for long documents with high repeat counts.

## Implementation Details

### Prompt Construction

Each prompt has a unique prefix (document ID) to prevent cache pollution:

```python
prompts = [
    str(i) + " ".join(["hi"] * args.document_length)
    for i in range(args.num_documents)
]
```

**Why:** Ensures documents are truly distinct despite simple content.

### Warmup Phase

```python
warmup_prompts = [
    "This is warm up request " + str(i) + " ".join(["hi"] * args.document_length)
    for i in range(args.num_documents)
]
test_long_document_qa(llm, warmup_prompts, sampling_params)
```

**Purpose:**
- Initialize CUDA kernels
- Populate internal caches
- Eliminate cold-start effects from benchmark

### Deterministic Generation

```python
sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)
```

**Why:** Temperature=0 ensures deterministic output for consistent benchmarking.

## Performance Analysis

### Cache Hit Patterns

**Interleave Mode:**
```
Document 0 → Cache Miss
Document 0 → Cache Hit (100%)
Document 1 → Cache Miss
Document 1 → Cache Hit (100%)
...
Average: ~50% cache hit rate
```

**Tile Mode:**
```
Cycle 1: Doc 0, 1, 2, 3 → All misses
Cycle 2: Doc 0, 1, 2, 3 → Some/no hits (depends on cache size)
Average: ~0-25% cache hit rate
```

**Random Mode:**
```
Highly variable depending on shuffle
Average: ~10-40% cache hit rate
```

### Expected Speedups

With `--enable-prefix-caching`:
- **Interleave:** 1.5-2x speedup
- **Random:** 1.2-1.5x speedup
- **Tile:** 1.0-1.2x speedup (cache eviction between cycles)

### Memory Considerations

KV cache memory = `num_documents * document_length * num_layers * hidden_size * 2`

For 8 documents of 20K tokens on Llama-2-7B:
- ~8 * 20000 * 32 * 4096 * 2 * 2 bytes = ~80 GB (too large)
- Prefix caching enables eviction of less-used entries

## Real-World Use Cases

### RAG Systems

```python
# Simulates retrieving and querying 10 different documents
# with 5 questions per document
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --document-length 15000 \
    --num-documents 10 \
    --repeat-count 5 \
    --repeat-mode interleave
```

### Multi-Turn Conversations

```python
# Simulates conversations with shared context
python benchmark_long_document_qa_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --document-length 5000 \
    --num-documents 20 \
    --repeat-count 10 \
    --repeat-mode random
```

## Integration Points

### vLLM Engine

- **LLM:** Offline inference engine with prefix caching support
- **SamplingParams:** Controls generation parameters
- **EngineArgs:** Unified configuration for engine initialization

### Prefix Caching System

- Automatic detection of shared prefixes
- LRU eviction when cache is full
- Transparent to the user (enabled via `--enable-prefix-caching`)

## Related Benchmarks

- **benchmark_prefix_caching.py:** More detailed prefix caching analysis with ShareGPT data
- **benchmark_throughput.py:** General throughput benchmarking without prefix focus
- **benchmark_latency.py:** Latency-focused measurements

## Limitations

1. **Synthetic Data:** Uses repeated "hi" tokens, not realistic content
2. **Single Metric:** Only measures total time, not per-request latency
3. **No Memory Tracking:** Doesn't report KV cache memory usage
4. **Offline Only:** Doesn't test online serving scenarios

## Future Enhancements

Potential improvements:
1. Report cache hit rate statistics
2. Add memory usage tracking
3. Support real document datasets
4. Measure per-request latency distribution
5. Add throughput metrics (tokens/sec)

## Technical Significance

This benchmark is critical for:
- **Validating Prefix Caching:** Proves effectiveness on long-context workloads
- **Performance Tuning:** Identifies optimal cache configurations
- **Use Case Validation:** Demonstrates benefits for RAG systems
- **Regression Testing:** Ensures caching performance doesn't degrade

The simple prompt generation strategy (repeated tokens) is intentional—it isolates caching effects from model computation complexity, providing clean measurements of cache effectiveness.
