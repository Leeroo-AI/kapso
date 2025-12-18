{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Request Tracking]], [[domain::Metrics]], [[domain::Intermediate Tensors]], [[domain::Pipeline Parallelism]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
The sequence module provides core data structures for tracking request metrics and intermediate tensors in vLLM's inference pipeline.

=== Description ===
This module defines fundamental dataclasses used throughout vLLM for request lifecycle tracking and pipeline parallelism. Key components include:

* '''RequestMetrics:''' Captures timing information for requests from arrival through completion
* '''IntermediateTensors:''' Stores hidden states and residuals passed between pipeline stages
* '''Token constants:''' Defines special token IDs and array types for token handling

The IntermediateTensors class is specially designed to work with PyTorch Dynamo compilation while providing dictionary-like access to tensors. RequestMetrics tracks detailed timing for performance analysis and monitoring.

=== Usage ===
Use these classes when you need to:
* Track request latency and throughput metrics
* Pass intermediate activations between pipeline stages
* Measure scheduler, model forward, and execution times
* Debug performance bottlenecks in inference pipeline
* Implement custom pipeline parallelism strategies
* Handle KV connector outputs in multi-stage pipelines

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/sequence.py vllm/sequence.py]

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class RequestMetrics:
    arrival_time: float
    last_token_time: float
    first_scheduled_time: float | None
    first_token_time: float | None
    time_in_queue: float | None
    finished_time: float | None = None
    scheduler_time: float | None = None
    model_forward_time: float | None = None
    model_execute_time: float | None = None

@dataclass
class IntermediateTensors:
    tensors: dict[str, torch.Tensor]
    kv_connector_output: KVConnectorOutput | None

    def __getitem__(self, key: str | slice) -> torch.Tensor | IntermediateTensors
    def __setitem__(self, key: str, value: torch.Tensor) -> None
    def items() -> dict_items
    def __len__() -> int
    def __eq__(other: object) -> bool

# Constants
VLLM_TOKEN_ID_ARRAY_TYPE = "l"
VLLM_INVALID_TOKEN_ID = -1
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.sequence import (
    RequestMetrics,
    IntermediateTensors,
    VLLM_TOKEN_ID_ARRAY_TYPE,
    VLLM_INVALID_TOKEN_ID,
)
</syntaxhighlight>

== I/O Contract ==

=== RequestMetrics Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| arrival_time || float || Timestamp when request arrived
|-
| last_token_time || float || Timestamp of last token generation
|-
| first_scheduled_time || float &#124; None || When request was first scheduled
|-
| first_token_time || float &#124; None || When first token was generated
|-
| time_in_queue || float &#124; None || Time spent waiting in queue
|-
| finished_time || float &#124; None || When request completed
|-
| scheduler_time || float &#124; None || Total time in scheduler
|-
| model_forward_time || float &#124; None || Total model forward pass time
|-
| model_execute_time || float &#124; None || Total model execute time
|}

=== IntermediateTensors Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tensors || dict[str, torch.Tensor] || Named tensors (hidden states, residuals)
|-
| kv_connector_output || KVConnectorOutput &#124; None || KV cache connector output for pipeline stage
|}

== Usage Examples ==

=== Track Request Metrics ===
<syntaxhighlight lang="python">
import time
from vllm.sequence import RequestMetrics

# Create metrics at request arrival
arrival = time.time()
metrics = RequestMetrics(
    arrival_time=arrival,
    last_token_time=arrival,
    first_scheduled_time=None,
    first_token_time=None,
    time_in_queue=None,
)

# Update when scheduled
schedule_time = time.time()
metrics.first_scheduled_time = schedule_time
metrics.time_in_queue = schedule_time - arrival

# Update when first token generated
first_token = time.time()
metrics.first_token_time = first_token

# Update timing metrics
metrics.scheduler_time = 0.05  # 50ms in scheduler
metrics.model_forward_time = 0.1  # 100ms forward pass
metrics.model_execute_time = 0.15  # 150ms total execute

# Mark completion
metrics.finished_time = time.time()

# Calculate end-to-end latency
e2e_latency = metrics.finished_time - metrics.arrival_time
time_to_first_token = metrics.first_token_time - metrics.arrival_time

print(f"E2E latency: {e2e_latency:.3f}s")
print(f"TTFT: {time_to_first_token:.3f}s")
print(f"Time in queue: {metrics.time_in_queue:.3f}s")
</syntaxhighlight>

=== Create and Use IntermediateTensors ===
<syntaxhighlight lang="python">
import torch
from vllm.sequence import IntermediateTensors

# Create intermediate tensors for pipeline stage
hidden_states = torch.randn(32, 128, 4096)
residuals = torch.randn(32, 128, 4096)

intermediate = IntermediateTensors(
    tensors={
        "hidden_states": hidden_states,
        "residual": residuals,
    },
    kv_connector_output=None
)

# Dictionary-like access
h = intermediate["hidden_states"]
print(h.shape)  # torch.Size([32, 128, 4096])

# Set new tensor
intermediate["attention_mask"] = torch.ones(32, 128)

# Iterate over tensors
for name, tensor in intermediate.items():
    print(f"{name}: {tensor.shape}")

# Check length
print(len(intermediate))  # 3

# Slice across batch dimension
batch_slice = intermediate[:16]
print(batch_slice["hidden_states"].shape)  # torch.Size([16, 128, 4096])
</syntaxhighlight>

=== Pipeline Stage Communication ===
<syntaxhighlight lang="python">
import torch
from vllm.sequence import IntermediateTensors

# Stage 1: Embedding + early layers
def pipeline_stage_1(input_ids):
    embeddings = embed_tokens(input_ids)
    hidden = early_layers(embeddings)

    return IntermediateTensors(
        tensors={
            "hidden_states": hidden,
            "residual": hidden.clone(),
        }
    )

# Stage 2: Middle layers
def pipeline_stage_2(intermediate: IntermediateTensors):
    hidden = intermediate["hidden_states"]
    residual = intermediate["residual"]

    hidden = middle_layers(hidden, residual)

    return IntermediateTensors(
        tensors={
            "hidden_states": hidden,
            "residual": residual,
        }
    )

# Stage 3: Final layers + output
def pipeline_stage_3(intermediate: IntermediateTensors):
    hidden = intermediate["hidden_states"]
    logits = final_layers(hidden)
    return logits

# Execute pipeline
input_ids = torch.randint(0, 50000, (32, 128))
stage1_out = pipeline_stage_1(input_ids)
stage2_out = pipeline_stage_2(stage1_out)
logits = pipeline_stage_3(stage2_out)
</syntaxhighlight>

=== Compare IntermediateTensors ===
<syntaxhighlight lang="python">
import torch
from vllm.sequence import IntermediateTensors

tensor1 = torch.randn(10, 20)
tensor2 = torch.randn(10, 20)

it1 = IntermediateTensors(tensors={"hidden": tensor1})
it2 = IntermediateTensors(tensors={"hidden": tensor1})  # Same tensor
it3 = IntermediateTensors(tensors={"hidden": tensor2})  # Different values

print(it1 == it2)  # True (same tensor data)
print(it1 == it3)  # False (different values)

# Different keys
it4 = IntermediateTensors(tensors={"other": tensor1})
print(it1 == it4)  # False (different keys)
</syntaxhighlight>

=== Use Token Constants ===
<syntaxhighlight lang="python">
import array
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, VLLM_INVALID_TOKEN_ID

# Create efficient token ID array
token_ids = array.array(VLLM_TOKEN_ID_ARRAY_TYPE, [1, 2, 3, 4, 5])

# Mark invalid tokens
token_ids.append(VLLM_INVALID_TOKEN_ID)

# Filter valid tokens
valid_tokens = [t for t in token_ids if t != VLLM_INVALID_TOKEN_ID]
print(valid_tokens)  # [1, 2, 3, 4, 5]
</syntaxhighlight>

=== Calculate Performance Metrics ===
<syntaxhighlight lang="python">
from vllm.sequence import RequestMetrics

def analyze_metrics(metrics: RequestMetrics):
    """Analyze request performance metrics."""

    # Time to first token (prefill latency)
    ttft = metrics.first_token_time - metrics.arrival_time

    # End-to-end latency
    e2e = metrics.finished_time - metrics.arrival_time

    # Decode latency (excluding prefill)
    decode = e2e - ttft

    # Queueing overhead
    queue_ratio = metrics.time_in_queue / e2e

    # Model execution efficiency
    exec_ratio = metrics.model_execute_time / e2e

    return {
        "ttft_ms": ttft * 1000,
        "e2e_ms": e2e * 1000,
        "decode_ms": decode * 1000,
        "queue_ratio": queue_ratio,
        "exec_ratio": exec_ratio,
        "throughput_tokens_per_sec": num_tokens / e2e,
    }

# Example usage
metrics = RequestMetrics(
    arrival_time=0.0,
    last_token_time=1.5,
    first_scheduled_time=0.05,
    first_token_time=0.2,
    time_in_queue=0.05,
    finished_time=1.5,
    scheduler_time=0.1,
    model_forward_time=1.0,
    model_execute_time=1.3,
)

stats = analyze_metrics(metrics)
print(f"TTFT: {stats['ttft_ms']:.1f}ms")
print(f"E2E: {stats['e2e_ms']:.1f}ms")
print(f"Queue overhead: {stats['queue_ratio']*100:.1f}%")
</syntaxhighlight>

=== Working with KV Connector ===
<syntaxhighlight lang="python">
from vllm.sequence import IntermediateTensors

# Stage with KV connector output
def stage_with_kv_connector(input_data):
    hidden = process_layer(input_data)

    # Simulate KV connector output
    kv_output = {
        "kv_cache_page_indices": torch.tensor([0, 1, 2]),
        "kv_cache_block_offsets": torch.tensor([0, 512, 1024]),
    }

    return IntermediateTensors(
        tensors={"hidden_states": hidden},
        kv_connector_output=kv_output
    )

# Next stage uses KV connector info
def next_stage(intermediate: IntermediateTensors):
    hidden = intermediate["hidden_states"]
    kv_info = intermediate.kv_connector_output

    # Use KV cache info for efficient attention
    if kv_info:
        output = efficient_attention(hidden, kv_info)
    else:
        output = standard_attention(hidden)

    return output
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[Pipeline Parallelism]]
* [[Request Scheduling]]
* [[Performance Monitoring]]
* [[KV Cache Management]]
