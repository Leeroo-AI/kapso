# PrefixCachingBenchmark - Automatic Prefix Caching Efficiency Benchmark

## Overview

**File:** `/tmp/praxium_repo_583nq7ea/benchmarks/benchmark_prefix_caching.py` (277 lines)

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Purpose:** Comprehensive benchmark for automatic prefix caching using realistic (ShareGPT) or synthetic prompts with configurable prefix lengths and repetition patterns, measuring KV cache reuse benefits.

## Core Architecture

### Request Dataclass

**Lines:** 59-63

```python
@dataclass
class Request:
    prompt: str
    prompt_len: int
    output_len: int
```

Simple container for request metadata used throughout sampling and filtering.

### Main Test Function

**Lines:** 50-57

```python
def test_prefix(llm=None, sampling_params=None, prompts=None):
    start_time = time.time()
    llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    print(f"cost time {end_time - start_time}")
```

**Metric:** Total generation time (simpler than long_document_qa version).

## Prompt Sampling Strategies

### Random Token Sampling

**Lines:** 66-74

```python
def sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)

    # Remove special tokens (BOS, EOS, PAD, etc.)
    return random.choices(
        [v for v in vocab.values() if v not in all_special_ids],
        k=length,
    )
```

**Purpose:** Generate random but valid token sequences for synthetic prompts.

### Dataset-Based Sampling

**Lines:** 77-123

```python
def sample_requests_from_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
) -> list[Request]:

    # Load ShareGPT dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter conversations with >= 2 turns
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Extract first two turns (prompt, completion)
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    random.shuffle(dataset)

    min_len, max_len = input_length_range
    filtered_requests: list[Request] = []

    for i in range(len(dataset)):
        if len(filtered_requests) == num_requests:
            break

        # Tokenize prompt and completion
        prompt_token_ids = tokenizer(dataset[i][0]).input_ids
        prompt = tokenizer.decode(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len

        # Filter by length range
        if min_len <= prompt_len <= max_len:
            filtered_requests.append(Request(prompt, prompt_len, output_len))

    return filtered_requests
```

**Features:**
- Uses real conversational data (ShareGPT)
- Filters by token length range
- Preserves natural language structure
- Supports fixed or variable output lengths

### Synthetic Random Sampling

**Lines:** 126-148

```python
def sample_requests_from_random(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
    prefix_len: int,
) -> list[Request]:

    requests = []
    # Generate common prefix once
    prefix_token_ids = sample_tokens(tokenizer, prefix_len)
    min_len, max_len = input_length_range

    for i in range(num_requests):
        # Generate unique suffix
        unique_part_token_ids = sample_tokens(
            tokenizer, random.randint(min_len - prefix_len, max_len - prefix_len)
        )
        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)

        assert min_len <= prompt_len <= max_len
        requests.append(Request(prompt, prompt_len, fixed_output_len))

    return requests
```

**Key Feature:** All prompts share exact same `prefix_len` tokens, guaranteeing cache hits.

### Request Repetition and Sorting

**Lines:** 151-159

```python
def repeat_and_sort_requests(
    requests: list[Request], repeat_count: int, sort: bool = False
) -> list[str]:

    repeated_requests = requests * repeat_count

    if sort:
        repeated_requests.sort(key=lambda x: x[1])  # Sort by prompt length
    else:
        random.shuffle(repeated_requests)

    return [req.prompt for req in repeated_requests]
```

**Sorting Rationale:** Longer prompts first maximizes prefix overlap with subsequent shorter prompts.

## Main Execution Flow

**Lines:** 162-217

```python
def main(args):
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(":")))
    random.seed(args.seed)

    # Sample requests from dataset or random
    if args.dataset_path is not None:
        if args.prefix_len > 0:
            raise ValueError("prefix-len not supported with dataset-path")
        filtered_requests = sample_requests_from_dataset(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
        )
    else:
        filtered_requests = sample_requests_from_random(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
            prefix_len=args.prefix_len,
        )

    # Print statistics
    prompt_lens = [req.prompt_len for req in filtered_requests]
    print(f"Sampled {len(filtered_requests)} requests.")
    print(f"Average input length: {sum(prompt_lens) / len(prompt_lens)}")
    print(f"P50 input length: {sorted(prompt_lens)[len(prompt_lens) // 2]}")
    print(f"Min Prompt Length: {min(prompt_lens)}")
    print(f"Max Prompt Length: {max(prompt_lens)}")

    # Initialize engine
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )

    # Repeat and optionally sort
    prompts = repeat_and_sort_requests(
        filtered_requests, repeat_count=args.repeat_count, sort=args.sort
    )

    print("------start generating------")
    test_prefix(llm=llm, prompts=prompts, sampling_params=sampling_params)
```

## Command-Line Arguments

**Lines:** 220-271

```python
def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark prefix caching performance"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to ShareGPT dataset"
    )

    parser.add_argument("--output-len", type=int, default=10)

    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of prompts to sample"
    )

    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Repeat each prompt N times"
    )

    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort prompts by input length (longest first)"
    )

    parser.add_argument(
        "--input-length-range",
        type=str,
        required=True,
        help='Format: "min:max" (e.g., "128:256")'
    )

    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Common prefix length (only for synthetic prompts)"
    )

    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help="Skip detokenization for faster benchmarking"
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser
```

## Usage Examples

### Basic Synthetic Benchmark

```bash
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-prompts 1 \
    --repeat-count 100 \
    --input-length-range 128:256
```

**Behavior:** Single prompt repeated 100 times, perfect cache hits after first.

### ShareGPT Realistic Workload

```bash
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --enable-prefix-caching \
    --num-prompts 20 \
    --repeat-count 5 \
    --input-length-range 128:256
```

**Behavior:** 20 diverse prompts, each repeated 5 times, tests cache with natural language.

### Synthetic with Shared Prefix

```bash
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-prompts 50 \
    --repeat-count 2 \
    --input-length-range 512:1024 \
    --prefix-len 256
```

**Behavior:** 50 prompts, all sharing first 256 tokens, tests partial prefix caching.

### Sorted for Maximum Cache Hits

```bash
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-prompts 30 \
    --repeat-count 3 \
    --input-length-range 100:1000 \
    --sort
```

**Behavior:** Longest prompts first, maximizes overlap with subsequent shorter prompts.

### Performance Comparison

```bash
# With caching
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-prompts 20 \
    --repeat-count 10 \
    --input-length-range 256:512

# Without caching (baseline)
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --num-prompts 20 \
    --repeat-count 10 \
    --input-length-range 256:512
```

### Fast Benchmarking (Skip Detokenization)

```bash
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-prompts 100 \
    --repeat-count 5 \
    --input-length-range 128:256 \
    --disable-detokenize
```

## Implementation Details

### Token Filtering

```python
all_special_ids = set(tokenizer.all_special_ids)
return random.choices(
    [v for v in vocab.values() if v not in all_special_ids],
    k=length,
)
```

**Why:** Special tokens (BOS, EOS) can cause tokenization issues, so they're excluded from random sampling.

### Length Range Enforcement

```python
assert min_len <= prompt_len <= max_len, (
    f"prompt_len {prompt_len} out of range {min_len}:{max_len}"
)
```

**Rationale:** Ensures all prompts fall within specified range for consistent comparison.

### Statistics Reporting

```python
prompt_lens = [req.prompt_len for req in filtered_requests]
print(f"Average input length: {sum(prompt_lens) / len(prompt_lens)}")
print(f"P50 input length: {sorted(prompt_lens)[len(prompt_lens) // 2]}")
print(f"Min Prompt Length: {min(prompt_lens)}")
print(f"Max Prompt Length: {max(prompt_lens)}")
```

**Purpose:** Helps understand prompt distribution and validate sampling.

## Performance Analysis

### Expected Cache Hit Rates

**Synthetic with prefix_len=256, num_prompts=10, repeat_count=5:**
- First 10 requests: Partial hits (256 tokens cached)
- Remaining 40 requests: Full prefix + partial unique hits
- Effective cache hit rate: ~60-70%

**ShareGPT with repeat_count=5:**
- First occurrence: Cache miss
- Subsequent 4: Full cache hit (if not evicted)
- Effective cache hit rate: ~80% (assuming sufficient cache)

**Sorted mode:**
- Longest prompt: Cache miss
- Each shorter prompt: Prefix of longer prompts likely cached
- Effective cache hit rate: ~40-60% (depends on overlap)

### Memory Requirements

KV cache memory per token: `num_layers * hidden_size * 2 * sizeof(dtype)`

For Llama-2-7B (32 layers, 4096 hidden, fp16):
- Per token: 32 * 4096 * 2 * 2 = 524,288 bytes ≈ 512 KB
- For 100 unique prompts @ 512 tokens avg: ~25 GB KV cache

## Real-World Scenarios

### Chatbot with Common System Prompts

```bash
# System prompt as prefix, varying user messages
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-prompts 100 \
    --repeat-count 1 \
    --input-length-range 200:300 \
    --prefix-len 150  # System prompt
```

### Document Q&A

```bash
# Multiple questions about same documents
python benchmark_prefix_caching.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching \
    --num-prompts 10 \
    --repeat-count 20 \
    --input-length-range 2000:3000 \
    --prefix-len 1500  # Document context
```

## Integration Points

### Tokenizer

```python
try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer
```

**Fallback:** Handles different vLLM versions/installations.

### vLLM Engine

- **LLM:** Offline inference with automatic prefix caching
- **SamplingParams:** Supports `detokenize` parameter for perf tuning
- **EngineArgs:** Unified CLI argument handling

### ShareGPT Dataset

Standard format with conversations array:
```json
{
  "conversations": [
    {"value": "user message"},
    {"value": "assistant response"}
  ]
}
```

## Comparison with Other Benchmarks

### vs benchmark_long_document_qa_throughput.py

- **Long Doc:** Focuses on very long contexts (20K tokens)
- **Prefix Caching:** Broader range, more flexible sampling

### vs benchmark_throughput.py

- **Throughput:** General performance, no caching focus
- **Prefix Caching:** Specifically designed to measure cache benefits

## Limitations

1. **No Cache Hit Reporting:** Doesn't report actual cache hit rate
2. **Simple Sorting:** Only length-based, not content-aware
3. **No Memory Tracking:** Doesn't measure KV cache memory usage
4. **Single Run:** No statistical confidence intervals

## Future Enhancements

Potential improvements:
1. Add cache hit rate instrumentation
2. Support multiple random seeds for confidence intervals
3. Report memory usage statistics
4. Add support for more datasets (MMLU, C4, etc.)
5. Implement content-aware prefix detection

## Technical Significance

This benchmark is essential for:
- **Feature Validation:** Proves prefix caching works correctly
- **Performance Quantification:** Measures speedup under various conditions
- **Regression Prevention:** Catches caching performance degradation
- **Configuration Tuning:** Helps determine optimal cache sizes
- **User Education:** Demonstrates when to enable prefix caching

The flexibility of both dataset-based and synthetic sampling makes this benchmark suitable for both realistic evaluations and controlled experiments isolating specific caching behaviors.
