# BenchmarkUtils - Shared Benchmark Utilities

## Overview

**File:** `/tmp/praxium_repo_583nq7ea/benchmarks/benchmark_utils.py` (125 lines)

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Purpose:** Provides shared utilities for all vLLM benchmarks including high-precision timing (TimeCollector), JSON serialization with special value handling (InfEncoder), and PyTorch OSS benchmark format conversion for standardized result reporting.

## Core Components

### TimeCollector Class

**Lines:** 78-125

```python
class TimeCollector:
    """
    Collect time measurements and generate statistics.

    Example Usage:
        collector = TimeCollector(TimeCollector.US)
        for _ in range(total_iteration):
            with collector:
                ...  # timed code
        collector.dump_avg_max()
    """

    # Time scale constants
    NS: int = 1                # Nanoseconds
    US: int = NS * 1000        # Microseconds
    MS: int = US * 1000        # Milliseconds
    S: int = MS * 1000         # Seconds

    def __init__(self, scale: int) -> None:
        self.cnt: int = 0
        self._sum: int = 0
        self._max: int | None = None
        self.scale = scale
        self.start_time: int = time.monotonic_ns()

    def collect(self, v: int) -> None:
        """Collect a single measurement (in nanoseconds)"""
        self.cnt += 1
        self._sum += v
        if self._max is None:
            self._max = v
        else:
            self._max = max(self._max, v)

    def avg(self) -> float | str:
        """Return average in specified scale"""
        return self._sum * 1.0 / self.cnt / self.scale if self.cnt > 0 else "N/A"

    def max(self) -> float | str:
        """Return maximum in specified scale"""
        return self._max / self.scale if self._max else "N/A"

    def dump_avg_max(self) -> list[float | str]:
        """Return [average, maximum] as list"""
        return [self.avg(), self.max()]

    def __enter__(self) -> None:
        """Context manager entry: start timing"""
        self.start_time = time.monotonic_ns()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Context manager exit: collect measurement"""
        self.collect(time.monotonic_ns() - self.start_time)
```

**Key Features:**
- **High Precision:** Uses `time.monotonic_ns()` for nanosecond timing
- **Context Manager:** Clean `with` statement usage
- **Scale Conversion:** Automatic conversion to US/MS/S
- **Statistics:** Tracks count, sum, max for avg/max calculation

### InfEncoder Class

**Lines:** 54-66

```python
class InfEncoder(json.JSONEncoder):
    """JSON encoder that handles infinity and NaN values"""

    def clear_inf(self, o: Any):
        """Recursively convert inf to string 'inf'"""
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        """Override iterencode to preprocess objects"""
        return super().iterencode(self.clear_inf(o), *args, **kwargs)
```

**Purpose:** Standard `json.dump()` fails on `float('inf')` values, this encoder converts them to string representation.

### PyTorch Benchmark Format Conversion

**Lines:** 12-51

```python
def convert_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: dict[str, list], extra_info: dict[str, Any]
) -> list:
    """
    Convert benchmark results to PyTorch OSS benchmark format.
    Format: https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database

    One record per metric.
    """
    records = []

    # Only convert if environment variable is set
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),  # Convert namespace to dict
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        # Ensure tensor_parallel_size is included
        tp = record["benchmark"]["extra_info"]["args"].get("tensor_parallel_size")
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"]["tensor_parallel_size"] = (
                extra_info["tensor_parallel_size"]
            )

        records.append(record)

    return records
```

**Format Structure:**
```json
{
  "benchmark": {
    "name": "vLLM benchmark",
    "extra_info": {"args": {...}}
  },
  "model": {"name": "meta-llama/Llama-2-7b-chat-hf"},
  "metric": {
    "name": "throughput",
    "benchmark_values": [1234.56, 1245.67, ...],
    "extra_info": {"tensor_parallel_size": 4}
  }
}
```

### JSON Writing Utility

**Lines:** 68-76

```python
def write_to_json(filename: str, records: list) -> None:
    """Write records to JSON file with special value handling"""
    with open(filename, "w") as f:
        json.dump(
            records,
            f,
            cls=InfEncoder,
            default=lambda o: f"<{type(o).__name__} object is not JSON serializable>",
        )
```

**Features:**
- Uses `InfEncoder` for infinity handling
- Fallback for non-serializable objects (converts to type name string)

## Usage Patterns

### Basic Timing

```python
from benchmark_utils import TimeCollector

# Create collector for microsecond precision
collector = TimeCollector(TimeCollector.US)

for i in range(100):
    with collector:
        # Code to time
        model.forward(input_data)

print(f"Average: {collector.avg():.2f} us")
print(f"Max: {collector.max():.2f} us")
```

### Multiple Time Scales

```python
# Nanosecond precision
ns_collector = TimeCollector(TimeCollector.NS)

# Millisecond precision
ms_collector = TimeCollector(TimeCollector.MS)

# Second precision
s_collector = TimeCollector(TimeCollector.S)
```

### Manual Collection

```python
collector = TimeCollector(TimeCollector.MS)

start = time.monotonic_ns()
# ... work ...
end = time.monotonic_ns()

collector.collect(end - start)
```

### JSON Serialization with Infinity

```python
from benchmark_utils import write_to_json

results = {
    "throughput": [1234.5, 2345.6, float('inf')],  # inf handled
    "latency": [10.2, 15.3, float('nan')],         # nan passed through
}

write_to_json("results.json", [results])
```

### PyTorch Benchmark Integration

```python
from benchmark_utils import convert_to_pytorch_benchmark_format
import os

# Enable PyTorch format
os.environ["SAVE_TO_PYTORCH_BENCHMARK_FORMAT"] = "1"

metrics = {
    "throughput": [1234.5, 1245.6, 1256.7],
    "latency_p50": [10.2, 10.5, 10.3],
}

extra_info = {
    "tensor_parallel_size": 4,
    "batch_size": 32,
}

records = convert_to_pytorch_benchmark_format(args, metrics, extra_info)
write_to_json("pytorch_benchmark.json", records)
```

## Implementation Details

### Monotonic Time

```python
self.start_time: int = time.monotonic_ns()
```

**Why monotonic?**
- `time.time()` can go backwards (NTP adjustments)
- `time.monotonic_ns()` always increases
- Essential for accurate duration measurements

### Nanosecond Storage

```python
self._sum: int = 0  # Store in nanoseconds
self._max: int | None = None
```

**Benefits:**
- High precision (1ns resolution)
- No floating point rounding errors
- Convert to desired scale on output

### Scale Conversion

```python
def avg(self) -> float | str:
    return self._sum * 1.0 / self.cnt / self.scale if self.cnt > 0 else "N/A"
```

**Example:**
- Stored: 1,234,567 ns
- Scale: TimeCollector.MS (1,000,000)
- Output: 1.234567 ms

### Infinity Handling

```python
elif isinstance(o, float) and math.isinf(o):
    return "inf"
```

**Why needed?**
- Division by zero can produce `float('inf')`
- `json.dumps()` raises `ValueError` on infinity
- Converting to string allows serialization

### Non-Serializable Objects

```python
default=lambda o: f"<{type(o).__name__} object is not JSON serializable>"
```

**Graceful Degradation:** Instead of crashing, saves type information.

## Real-World Usage

### In benchmark_throughput.py

```python
from benchmark_utils import TimeCollector

collector = TimeCollector(TimeCollector.S)

with collector:
    outputs = llm.generate(prompts, sampling_params)

print(f"Throughput: {total_tokens / collector.avg():.2f} tokens/s")
```

### In benchmark_latency.py

```python
from benchmark_utils import write_to_json

results = {
    "elapsed_time": elapsed,
    "requests_per_second": num_requests / elapsed,
    "tokens_per_second": total_tokens / elapsed,
}

if args.output_json:
    write_to_json(args.output_json, [results])
```

### CI/CD Integration

```bash
#!/bin/bash
export SAVE_TO_PYTORCH_BENCHMARK_FORMAT=1

python benchmark_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --output-json results.json

# Upload to PyTorch OSS benchmark database
curl -X POST https://pytorch-benchmark.org/api/upload \
    -H "Content-Type: application/json" \
    -d @results.json
```

## Performance Considerations

### TimeCollector Overhead

- `time.monotonic_ns()`: ~100ns per call
- Context manager: ~200ns overhead
- Negligible for workloads > 1us

### Memory Usage

```python
self.cnt: int          # 8 bytes
self._sum: int         # 8 bytes
self._max: int | None  # 8-16 bytes
Total: ~24-32 bytes per collector
```

**Minimal:** Can create many collectors without concern.

### JSON Encoding

- `InfEncoder`: Recursive traversal adds ~10-20% overhead
- Only used for final result serialization (not hot path)

## Integration Points

### All vLLM Benchmarks

- **benchmark_throughput.py**
- **benchmark_latency.py**
- **benchmark_prefix_caching.py**
- **benchmark_long_document_qa_throughput.py**
- **benchmark_prioritization.py**

### PyTorch OSS Benchmark

- Database: https://pytorch-benchmark.org
- Wiki: https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database

## Testing Recommendations

### Unit Tests

```python
def test_time_collector():
    collector = TimeCollector(TimeCollector.MS)

    # Simulate 5 measurements of ~10ms
    for _ in range(5):
        with collector:
            time.sleep(0.01)

    assert 9 < collector.avg() < 11  # ~10ms
    assert collector.cnt == 5

def test_inf_encoder():
    data = {"value": float('inf')}
    result = json.dumps(data, cls=InfEncoder)
    assert '"value": "inf"' in result
```

### Integration Tests

```python
def test_benchmark_format():
    args = argparse.Namespace(model="test-model")
    metrics = {"throughput": [1.0, 2.0]}
    extra_info = {"tp": 2}

    records = convert_to_pytorch_benchmark_format(args, metrics, extra_info)

    assert len(records) == 1
    assert records[0]["model"]["name"] == "test-model"
```

## Related Components

- **time module:** Provides monotonic_ns()
- **json module:** Standard library JSON encoding
- **argparse.Namespace:** Argument container
- **PyTorch Benchmark Database:** Result storage

## Limitations

1. **No Median/Percentiles:** Only average and max
2. **No Standard Deviation:** Limited statistical metrics
3. **Binary Infinity:** Treats +inf and -inf the same
4. **No NaN Handling:** NaN values not converted

## Future Enhancements

Potential improvements:
1. Add median/percentile calculations
2. Track standard deviation
3. Support more time sources (CUDA events)
4. Handle NaN values
5. Add histogram support

## Technical Significance

This module is foundational for benchmarking:
- **Consistency:** All benchmarks use same timing methodology
- **Accuracy:** Nanosecond precision eliminates measurement noise
- **Interoperability:** PyTorch format enables cross-project comparison
- **Robustness:** Graceful handling of special values prevents crashes

The TimeCollector pattern (context manager + statistics) is particularly elegant, combining convenience with precise measurement. The PyTorch OSS benchmark format integration positions vLLM benchmarks as first-class citizens in the broader ML performance ecosystem.
