# File: `vllm/sequence.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 98 |
| Classes | `RequestMetrics`, `IntermediateTensors` |
| Imports | dataclasses, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Request metrics and intermediate tensors

**Mechanism:** Defines RequestMetrics dataclass tracking timing information for requests (arrival_time, first_scheduled_time, first_token_time, queue_time, scheduler_time, model_forward_time, model_execute_time, finished_time). IntermediateTensors wraps hidden states and residuals passed between pipeline stages with dict-like interface and support for KV connector outputs. Both are fundamental data structures used throughout request processing.

**Significance:** RequestMetrics provides essential observability data for monitoring and optimizing serving performance. Timing breakdowns help identify bottlenecks in queue management, scheduling, and model execution. IntermediateTensors enables pipeline parallelism by packaging intermediate activations for cross-stage communication. Critical for understanding system behavior and enabling distributed inference.
