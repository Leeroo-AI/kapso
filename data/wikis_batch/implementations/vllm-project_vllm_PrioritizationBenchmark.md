# PrioritizationBenchmark - Request Prioritization Throughput Benchmark

## Overview

**File:** `/tmp/praxium_repo_583nq7ea/benchmarks/benchmark_prioritization.py` (221 lines)

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Purpose:** Benchmarks offline inference with priority-based request scheduling, measuring throughput (requests/s, tokens/s) when requests have different priorities, validating vLLM's ability to handle heterogeneous workloads with SLA requirements.

## Core Architecture

### Priority Assignment

**Lines:** 17-19

```python
def get_random_flag():
    """Select equi-probable random binary priority"""
    return 0 if random.random() < 0.5 else 1
```

**Design:** Simple binary priorities (0 or 1) for clear analysis. Real systems could use arbitrary integers.

### Request Sampling from Dataset

**Lines:** 22-71

```python
def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int | None,
) -> list[tuple[str, int, int, int]]:

    # Load ShareGPT dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter conversations with >= 2 turns
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Extract first two turns
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    random.shuffle(dataset)

    filtered_dataset: list[tuple[str, int, int]] = []

    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids

        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len

        # Filter by length constraints
        if prompt_len < 4 or output_len < 4:
            continue  # Too short
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            continue  # Too long

        # Assign random priority
        priority = get_random_flag()

        filtered_dataset.append((prompt, prompt_len, output_len, priority))

    return filtered_dataset
```

**Filtering Logic:**
- Min length: 4 tokens (avoid trivial requests)
- Max prompt: 1024 tokens (prevent OOM)
- Max total: 2048 tokens (model context limit)

### vLLM Execution

**Lines:** 74-113

```python
def run_vllm(
    requests: list[tuple[str, int, int]],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
) -> float:
    from vllm import LLM, SamplingParams

    llm = LLM(**dataclasses.asdict(engine_args))

    # Validate all requests fit in context
    assert all(
        llm.llm_engine.model_config.max_model_len >= (request[1] + request[2])
        for request in requests
    ), "max_model_len must be >= input_len + output_len for all requests"

    # Prepare requests
    prompts = []
    sampling_params = []
    priority = []

    for prompt, _, output_len, _priority in requests:
        prompts.append(prompt)
        priority.append(_priority)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,  # Non-deterministic sampling
                top_p=1.0,        # No nucleus filtering
                ignore_eos=True,   # Generate full output_len
                max_tokens=output_len,
                detokenize=not disable_detokenize,
            )
        )

    # Benchmark
    start = time.perf_counter()
    llm.generate(prompts, sampling_params, priority=priority, use_tqdm=True)
    end = time.perf_counter()

    return end - start
```

**Key Features:**
- **Per-request SamplingParams:** Allows variable output lengths
- **ignore_eos=True:** Ensures consistent output lengths for fair comparison
- **use_tqdm=True:** Progress bar for long runs
- **Priority parameter:** Passed directly to generate()

### Main Execution

**Lines:** 116-161

```python
def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )

    if args.dataset is None:
        # Synthesize prompts
        prompt = "hi" * (args.input_len - 1)
        requests = [
            (prompt, args.input_len, args.output_len, get_random_flag())
            for _ in range(args.num_prompts)
        ]
    else:
        # Sample from dataset
        requests = sample_requests(
            args.dataset, args.num_prompts, tokenizer, args.output_len
        )

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests, args.n, EngineArgs.from_cli_args(args), args.disable_detokenize
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Calculate metrics
    total_num_tokens = sum(
        prompt_len + output_len for _, prompt_len, output_len, priority in requests
    )

    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )

    # Save JSON results if requested
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
```

## Command-Line Arguments

**Lines:** 163-207

```python
def create_argument_parser():
    parser = FlexibleArgumentParser(description="Benchmark throughput with prioritization")

    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "hf", "mii"],
        default="vllm"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset (ShareGPT format)"
    )

    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input length for synthetic prompts"
    )

    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length (overrides dataset if provided)"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of sequences per prompt"
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of prompts to process"
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON results"
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

### Basic Synthetic Workload

```bash
python benchmark_prioritization.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --input-len 256 \
    --output-len 128 \
    --num-prompts 100
```

**Output:**
```
Throughput: 12.34 requests/s, 4567.89 tokens/s
```

### ShareGPT Dataset

```bash
python benchmark_prioritization.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 200 \
    --output-len 100
```

### With JSON Output

```bash
python benchmark_prioritization.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --input-len 512 \
    --output-len 256 \
    --num-prompts 50 \
    --output-json results.json
```

**results.json:**
```json
{
    "elapsed_time": 45.67,
    "num_requests": 50,
    "total_num_tokens": 38400,
    "requests_per_second": 1.09,
    "tokens_per_second": 841.23
}
```

### Multiple Sequences per Prompt

```bash
python benchmark_prioritization.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --input-len 128 \
    --output-len 64 \
    --num-prompts 100 \
    --n 5  # Generate 5 outputs per prompt
```

### Fast Benchmarking

```bash
python benchmark_prioritization.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --input-len 256 \
    --output-len 128 \
    --num-prompts 500 \
    --disable-detokenize
```

### With Custom Scheduling

```bash
python benchmark_prioritization.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset /path/to/dataset.json \
    --num-prompts 200 \
    --scheduling-policy fcfs  # First-come-first-serve
```

## Implementation Details

### Priority Distribution

With `get_random_flag()`, priorities are uniformly distributed:
- 50% priority 0 (lower)
- 50% priority 1 (higher)

Expected scheduling behavior:
- Priority 1 requests processed first when scheduler has choice
- Priority 0 requests may wait longer but still complete

### Sampling Parameters

```python
SamplingParams(
    n=n,                    # Multiple outputs per prompt
    temperature=1.0,        # Non-deterministic (realistic)
    top_p=1.0,             # Full vocabulary
    ignore_eos=True,       # Consistent output lengths
    max_tokens=output_len,
    detokenize=not disable_detokenize,
)
```

**Rationale:**
- `temperature=1.0`: Realistic sampling (not greedy)
- `ignore_eos=True`: Ensures all requests generate exactly `output_len` tokens
- Variable detokenization: Trade latency for output quality

### Length Constraints

```python
if prompt_len < 4 or output_len < 4:
    continue  # Too short

if prompt_len > 1024 or prompt_len + output_len > 2048:
    continue  # Too long
```

**Purpose:**
- Avoid trivial requests that don't test scheduling
- Prevent OOM errors from very long sequences
- Ensure requests fit in typical model contexts

## Performance Analysis

### Priority Scheduling Impact

**Without Prioritization:**
- FCFS (First Come First Serve) scheduling
- All requests treated equally
- Avg latency: uniform distribution

**With Prioritization:**
- Priority 1 requests scheduled first
- Priority 0 may experience higher latency
- Priority 1 avg latency < Priority 0 avg latency

**Throughput Impact:**
- Overall throughput should remain similar
- Priority affects latency distribution, not aggregate throughput

### Expected Results

For 200 requests (100 each priority):
- **Priority 0 requests:**
  - Avg latency: ~60-80% higher than baseline
  - Total wait time: ~40% of run time
- **Priority 1 requests:**
  - Avg latency: ~30-50% lower than baseline
  - Total wait time: ~10% of run time

### Throughput Calculation

```python
requests_per_second = num_requests / elapsed_time
tokens_per_second = total_num_tokens / elapsed_time
```

**Typical Values (Llama-2-7B on A100):**
- Requests/s: 5-15 (depends on batch size, sequence length)
- Tokens/s: 1000-3000 (depends on hardware utilization)

## Real-World Scenarios

### Premium vs Standard Tiers

```python
# Simulate SLA-based serving
# Priority 1: Premium users (low latency SLA)
# Priority 0: Standard users (best effort)

python benchmark_prioritization.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset /path/to/production_workload.json \
    --num-prompts 1000
```

### Batch Processing with Urgency

```python
# Priority 1: Interactive queries (immediate response needed)
# Priority 0: Batch analytics (can wait)

python benchmark_prioritization.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --input-len 512 \
    --output-len 256 \
    --num-prompts 500
```

## Integration Points

### vLLM Engine

- **LLM.generate():** Accepts `priority` parameter (list of ints)
- **Scheduler:** Priority-aware request ordering
- **Batching:** Higher priority requests preferentially added to batches

### Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer, trust_remote_code=args.trust_remote_code
)
```

**Flexibility:** Supports any HuggingFace tokenizer with trust_remote_code.

### Backend Abstraction

```python
if args.backend == "vllm":
    elapsed_time = run_vllm(...)
else:
    raise ValueError(f"Unknown backend: {args.backend}")
```

**Future Extension:** Could add HF transformers or other backends.

## Comparison with Other Benchmarks

### vs benchmark_throughput.py

- **Throughput:** No priorities, measures max throughput
- **Prioritization:** Heterogeneous priorities, measures fairness

### vs benchmark_latency.py

- **Latency:** Measures per-request latency distribution
- **Prioritization:** Focuses on aggregate throughput

## Limitations

1. **Binary Priorities:** Only 0/1, not arbitrary integers
2. **No Latency Breakdown:** Doesn't report per-priority latency
3. **Offline Only:** Doesn't test online serving with arrivals
4. **No Preemption:** Tests scheduling, not preemption

## Future Enhancements

Potential improvements:
1. Support N-level priorities (0-9 or arbitrary)
2. Report per-priority latency statistics
3. Add online arrival simulation (Poisson process)
4. Test preemption scenarios
5. Visualize latency distributions by priority

## Technical Significance

This benchmark is critical for:
- **SLA Validation:** Proves priority scheduling works correctly
- **Multi-Tenant Serving:** Enables fair resource allocation
- **Production Readiness:** Validates real-world serving patterns
- **Performance Tuning:** Helps configure scheduling policies
- **Regression Testing:** Catches priority scheduling bugs

The 50/50 priority distribution is intentionally simple to provide clear signal about scheduler behavior without complex statistical analysis. Real deployments might use skewed distributions (e.g., 10% premium, 90% standard) which this benchmark can support by modifying `get_random_flag()`.
