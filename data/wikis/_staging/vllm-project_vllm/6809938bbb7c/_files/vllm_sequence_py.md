# File: `vllm/sequence.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 98 |
| Classes | `RequestMetrics`, `IntermediateTensors` |
| Imports | dataclasses, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core data structures for tracking request metrics and intermediate states.

**Mechanism:** `RequestMetrics` dataclass captures timing information for a request: arrival time, first scheduled/token times, queue time, finished time, and execution times (scheduler, model forward, model execute). Used for performance monitoring and SLA tracking. `IntermediateTensors` holds hidden states and residuals passed between pipeline stages in pipeline-parallel execution. It wraps a dict of tensors and optional KV connector output, providing dict-like access with slicing support. The class is carefully designed to work with PyTorch Dynamo (avoiding eval'd strings).

**Significance:** Essential for observability and pipeline parallelism. Request metrics enable performance analysis, latency tracking, and identifying bottlenecks. This data feeds into monitoring systems and helps users understand inference performance. `IntermediateTensors` is critical for pipeline-parallel inference where intermediate activations must be passed between stages efficiently. The Dynamo-compatible design ensures these structures work with torch.compile optimization.
