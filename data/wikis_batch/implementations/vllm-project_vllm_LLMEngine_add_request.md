'''Metadata'''
{| class="wikitable"
|-
! Key !! Value
|-
| Knowledge Sources || vLLM Engine Source Code, Request Processing API
|-
| Domains || Request Scheduling, Multi-LoRA Batching
|-
| Last Updated || 2025-12-17
|}

== Overview ==

'''LLMEngine_add_request''' is the method for submitting inference requests to the vLLM engine, including requests with LoRA adapters. This method validates requests, processes inputs, and queues them for batched execution with appropriate adapter routing.

== Description ==

The LLMEngine.add_request() method serves as the primary entry point for adding generation requests to the engine's scheduling queue. When a request includes a LoRARequest parameter, the method coordinates with the LoRA management subsystem to ensure the specified adapter is available and properly routed.

=== Request Processing Flow ===

1. '''Input Processing''': Tokenizes prompt, processes multimodal inputs, validates parameters
2. '''LoRA Validation''': If lora_request provided, validates adapter ID and path
3. '''Request Creation''': Constructs internal EngineCoreRequest with all metadata
4. '''Adapter Loading''': Triggers async adapter loading if not in cache
5. '''Queue Insertion''': Adds request to scheduler queue with adapter association
6. '''State Tracking''': Registers request in output processor for result collection

=== LoRA-Specific Handling ===

When lora_request is provided:

* '''Adapter Resolution''': Engine checks if adapter_id is already loaded in GPU cache
* '''Cache Management''': If adapter not loaded and slots available, initiates background loading
* '''Slot Allocation''': If all slots full, request waits until a slot becomes available via eviction
* '''Batching Coordination''': Scheduler uses adapter_id for batching decisions
* '''Output Association''': Generated outputs include reference to original LoRARequest

The engine maintains internal mappings:
* adapter_id → loaded adapter weights
* adapter_id → list of active requests using that adapter
* adapter_id → cache slot position

=== Parallel Sampling Support ===

When sampling_params.n > 1, the engine creates multiple child requests:
* Each child has unique request_id but shares the parent's lora_request
* All children use the same adapter, ensuring consistent LoRA application
* Results are aggregated by output processor maintaining adapter context

== Code Reference ==

=== Source Location ===
* '''File''': vllm/v1/engine/llm_engine.py
* '''Class''': LLMEngine
* '''Method''': add_request

=== Signature ===

<syntaxhighlight lang="python">
def add_request(
    self,
    request_id: str,
    prompt: PromptType,
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    trace_headers: dict[str, str] | None = None,
    priority: int = 0,
) -> None:
    """
    Add a new inference request to the engine.

    Args:
        request_id: Unique identifier for this request
        prompt: Input prompt (str, token IDs, or multimodal dict)
        params: Sampling or pooling parameters
        arrival_time: Request arrival timestamp (default: current time)
        lora_request: Optional LoRA adapter to apply
        trace_headers: Optional tracing headers for observability
        priority: Request priority (higher = more important)
    """
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
</syntaxhighlight>

== I/O Contract ==

=== Input Parameters ===

{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| request_id || str || required || Unique identifier for request tracking
|-
| prompt || PromptType || required || Input text, token IDs, or multimodal dict
|-
| params || SamplingParams \| PoolingParams || required || Generation parameters
|-
| arrival_time || float \| None || None || Request timestamp (default: time.time())
|-
| lora_request || LoRARequest \| None || None || LoRA adapter to apply (None = base model)
|-
| trace_headers || dict[str, str] \| None || None || Distributed tracing headers
|-
| priority || int || 0 || Request priority for scheduling
|}

=== Output ===

{| class="wikitable"
|-
! Return Type !! Description
|-
| None || Request added asynchronously, results via step() or async iteration
|}

=== Side Effects ===

* Request added to internal scheduling queue
* Adapter loading initiated if needed (async background operation)
* Request state created in output processor
* Metrics updated (pending request count, adapter usage)

== Usage Examples ==

=== Example 1: Basic Request with LoRA ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

# Initialize engine
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_lora_rank=64
)
engine = LLMEngine.from_engine_args(engine_args)

# Create LoRA request
lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")

# Add request to engine
request_id = "req-001"
prompt = "Write a SQL query to select all users"
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

engine.add_request(
    request_id=request_id,
    prompt=prompt,
    params=sampling_params,
    lora_request=lora_request
)

# Process requests
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(f"Result: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 2: Multiple Adapters in Batch ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

# Engine supports 2 concurrent adapters
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=2,
    max_lora_rank=64
)
engine = LLMEngine.from_engine_args(engine_args)

# Define different adapters
sql_lora = LoRARequest("sql-lora", 1, "/path/to/sql-lora")
math_lora = LoRARequest("math-lora", 2, "/path/to/math-lora")

# Add requests with different adapters
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

engine.add_request("req-1", "SQL: Select users", sampling_params, lora_request=sql_lora)
engine.add_request("req-2", "Math: Solve x^2 + 5x + 6 = 0", sampling_params, lora_request=math_lora)
engine.add_request("req-3", "SQL: Join tables", sampling_params, lora_request=sql_lora)

# Engine batches requests intelligently
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            adapter_name = output.lora_request.name if output.lora_request else "base"
            print(f"[{adapter_name}] {output.request_id}: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 3: Base Model and LoRA Mixed ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=1
)
engine = LLMEngine.from_engine_args(engine_args)

lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")
sampling_params = SamplingParams(max_tokens=128)

# Some requests use LoRA, others use base model
engine.add_request("req-base-1", "General question", sampling_params, lora_request=None)
engine.add_request("req-lora-1", "SQL query", sampling_params, lora_request=lora_request)
engine.add_request("req-base-2", "Another general question", sampling_params, lora_request=None)

# Engine schedules appropriately
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(f"{output.request_id}: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 4: Dynamic Request Routing ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

# Initialize engine
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=3,
    max_lora_rank=64
)
engine = LLMEngine.from_engine_args(engine_args)

# Adapter registry
ADAPTERS = {
    "sql": LoRARequest("sql-lora", 1, "/path/to/sql-lora"),
    "code": LoRARequest("code-lora", 2, "/path/to/code-lora"),
    "math": LoRARequest("math-lora", 3, "/path/to/math-lora"),
}

# Route requests based on task type
def add_task_request(engine, task_type: str, prompt: str, request_id: str):
    lora_request = ADAPTERS.get(task_type)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    engine.add_request(
        request_id=request_id,
        prompt=prompt,
        params=sampling_params,
        lora_request=lora_request
    )

# Submit diverse tasks
add_task_request(engine, "sql", "SELECT * FROM users", "req-1")
add_task_request(engine, "code", "def fibonacci(n):", "req-2")
add_task_request(engine, "math", "Solve: 2x + 5 = 15", "req-3")

# Process all requests
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(f"{output.request_id}: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 5: Priority and Arrival Time ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest
import time

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=1
)
engine = LLMEngine.from_engine_args(engine_args)

lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")
sampling_params = SamplingParams(max_tokens=128)

# High priority request
engine.add_request(
    request_id="high-priority",
    prompt="Urgent SQL query",
    params=sampling_params,
    lora_request=lora_request,
    priority=10  # Higher priority
)

# Normal priority with explicit arrival time
engine.add_request(
    request_id="normal-priority",
    prompt="Regular SQL query",
    params=sampling_params,
    lora_request=lora_request,
    arrival_time=time.time(),
    priority=0  # Default priority
)

# High priority request should be processed first
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(f"Processed: {output.request_id}")
</syntaxhighlight>

=== Example 6: Parallel Sampling with LoRA ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=1
)
engine = LLMEngine.from_engine_args(engine_args)

lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")

# Generate multiple candidates with n>1
sampling_params = SamplingParams(
    n=3,  # Generate 3 different outputs
    temperature=0.8,
    max_tokens=128
)

engine.add_request(
    request_id="multi-sample",
    prompt="Write a SQL query",
    params=sampling_params,
    lora_request=lora_request
)

# All 3 samples use the same LoRA adapter
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(f"Request: {output.request_id}")
            for i, completion in enumerate(output.outputs):
                print(f"  Sample {i+1}: {completion.text}")
</syntaxhighlight>

=== Example 7: Error Handling ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=1
)
engine = LLMEngine.from_engine_args(engine_args)

# Invalid LoRA request (non-existent path)
try:
    invalid_lora = LoRARequest("bad-lora", 1, "/nonexistent/path")
    sampling_params = SamplingParams(max_tokens=128)

    engine.add_request(
        request_id="test",
        prompt="Test prompt",
        params=sampling_params,
        lora_request=invalid_lora
    )

    # Error may occur during adapter loading in step()
    while engine.has_unfinished_requests():
        outputs = engine.step()

except Exception as e:
    print(f"Error during LoRA inference: {e}")
</syntaxhighlight>

== Performance Considerations ==

* '''Cold Start''': First request for an adapter incurs loading overhead (100-500ms)
* '''Batching Efficiency''': Mixed-adapter batches are 10-30% slower than single-adapter batches
* '''Memory Pressure''': Each active adapter consumes GPU memory; monitor max_loras limits
* '''Scheduling Fairness''': Frequent adapter switching reduces throughput; batch similar requests

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_MultiLoRA_Inference]]
* [[related_to::vllm-project_vllm_LoRARequest]]
* [[related_to::vllm-project_vllm_LLMEngine_step]]
