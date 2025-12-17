'''Metadata'''
{| class="wikitable"
|-
! Key !! Value
|-
| Knowledge Sources || vLLM Output Structures, Response API
|-
| Domains || Output Formatting, Adapter Traceability
|-
| Last Updated || 2025-12-17
|}

== Overview ==

'''RequestOutput_lora''' is the output data structure that encapsulates generated results along with their associated LoRA adapter metadata. This implementation ensures complete traceability by including the LoRARequest in both RequestOutput and CompletionOutput objects.

== Description ==

The RequestOutput class serves as the primary container for inference results in vLLM. When a request is processed with a LoRA adapter, the RequestOutput instance includes a reference to the originating LoRARequest, enabling clients to identify which adapter produced the output.

=== Class Structure ===

RequestOutput contains:

* '''request_id''': Unique identifier matching the original request
* '''prompt''': Original input prompt text
* '''prompt_token_ids''': Tokenized prompt
* '''prompt_logprobs''': Log probabilities for prompt tokens (if requested)
* '''outputs''': List of CompletionOutput objects (one per completion)
* '''finished''': Boolean indicating if generation is complete
* '''metrics''': Request-level performance metrics
* '''lora_request''': LoRARequest object (None if base model used)
* '''encoder_prompt''': Encoder prompt for sequence-to-sequence models
* '''num_cached_tokens''': Prefix cache hit count

Each CompletionOutput within outputs also includes:

* '''lora_request''': Reference to the adapter used (same as parent RequestOutput)
* '''index''': Completion index (for n>1 sampling)
* '''text''': Generated text
* '''token_ids''': Generated token IDs
* '''cumulative_logprob''': Cumulative log probability
* '''logprobs''': Per-token log probabilities (if requested)
* '''finish_reason''': Why generation stopped
* '''stop_reason''': Specific stop token/string

=== Adapter Association Lifecycle ===

1. '''Request Submission''': LoRARequest passed to add_request()
2. '''Internal Tracking''': Engine maintains request → adapter mapping
3. '''Output Construction''': OutputProcessor creates RequestOutput with lora_request
4. '''Streaming Updates''': Each partial output includes adapter reference
5. '''Final Delivery''': Complete RequestOutput with full adapter metadata

The lora_request field is populated from the engine's internal state during output processing. When streaming, each partial RequestOutput maintains the same LoRARequest reference for consistency.

=== Access Patterns ===

Clients access adapter information via:

<syntaxhighlight lang="python">
# Top-level adapter access
if output.lora_request:
    adapter_name = output.lora_request.name
    adapter_id = output.lora_request.adapter_id

# Per-completion adapter access (n>1 case)
for completion in output.outputs:
    if completion.lora_request:
        print(f"Completion {completion.index} used {completion.lora_request.name}")
</syntaxhighlight>

== Code Reference ==

=== Source Location ===
* '''File''': vllm/outputs.py
* '''Classes''': RequestOutput, CompletionOutput

=== Signature ===

<syntaxhighlight lang="python">
@dataclass
class CompletionOutput:
    """The output data of one completion output of a request."""

    index: int
    text: str
    token_ids: GenericSequence[int]
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    lora_request: LoRARequest | None = None  # LoRA adapter used

class RequestOutput:
    """The output data of a completion request to the LLM."""

    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: PromptLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: RequestMetrics | RequestStateStats | None = None,
        lora_request: LoRARequest | None = None,  # LoRA adapter used
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request  # Adapter metadata
        # ... other fields
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.lora.request import LoRARequest
</syntaxhighlight>

== I/O Contract ==

=== Input (Constructor Parameters) ===

{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| request_id || str || Unique request identifier
|-
| prompt || str \| None || Original prompt text
|-
| prompt_token_ids || list[int] \| None || Tokenized prompt
|-
| prompt_logprobs || PromptLogprobs \| None || Prompt token log probabilities
|-
| outputs || list[CompletionOutput] || Generated completions (1+ items)
|-
| finished || bool || Whether generation is complete
|-
| metrics || RequestMetrics \| RequestStateStats \| None || Performance metrics
|-
| lora_request || LoRARequest \| None || Adapter used (None = base model)
|}

=== Output (Accessible Fields) ===

{| class="wikitable"
|-
! Field !! Type !! Description
|-
| request_id || str || Request identifier
|-
| outputs || list[CompletionOutput] || Generated results
|-
| finished || bool || Completion status
|-
| lora_request || LoRARequest \| None || Adapter metadata
|-
| metrics || RequestMetrics \| RequestStateStats \| None || Performance data
|}

== Usage Examples ==

=== Example 1: Basic Output with LoRA ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

# Initialize and add request
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_lora_rank=64
)
engine = LLMEngine.from_engine_args(engine_args)

lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

engine.add_request("req-1", "SQL query prompt", sampling_params, lora_request=lora_request)

# Process and check output
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            # Access adapter information
            if output.lora_request:
                print(f"Adapter: {output.lora_request.name}")
                print(f"Adapter ID: {output.lora_request.adapter_id}")
            else:
                print("Used base model")

            print(f"Generated: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 2: Comparing Multiple Adapters ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=3,
    max_lora_rank=64
)
engine = LLMEngine.from_engine_args(engine_args)

# Same prompt with different adapters
prompt = "Write a function to calculate factorial"
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

adapters = [
    LoRARequest("python-lora", 1, "/path/to/python-lora"),
    LoRARequest("cpp-lora", 2, "/path/to/cpp-lora"),
    LoRARequest("java-lora", 3, "/path/to/java-lora"),
]

# Submit requests
for i, lora_req in enumerate(adapters):
    engine.add_request(f"req-{i}", prompt, sampling_params, lora_request=lora_req)

# Collect results by adapter
results = {}
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            adapter_name = output.lora_request.name
            results[adapter_name] = output.outputs[0].text

# Compare outputs
for adapter_name, text in results.items():
    print(f"\n=== {adapter_name} ===")
    print(text)
</syntaxhighlight>

=== Example 3: Output Metrics per Adapter ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest
from collections import defaultdict

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=2
)
engine = LLMEngine.from_engine_args(engine_args)

# Track metrics per adapter
adapter_metrics = defaultdict(lambda: {"count": 0, "total_tokens": 0})

# Submit multiple requests
lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")
sampling_params = SamplingParams(max_tokens=128)

for i in range(10):
    engine.add_request(f"req-{i}", f"Prompt {i}", sampling_params, lora_request=lora_request)

# Collect metrics
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            adapter_name = output.lora_request.name if output.lora_request else "base"
            adapter_metrics[adapter_name]["count"] += 1

            # Count generated tokens
            for completion in output.outputs:
                adapter_metrics[adapter_name]["total_tokens"] += len(completion.token_ids)

# Report metrics
for adapter, metrics in adapter_metrics.items():
    print(f"\n{adapter}:")
    print(f"  Requests: {metrics['count']}")
    print(f"  Total tokens: {metrics['total_tokens']}")
    print(f"  Avg tokens/request: {metrics['total_tokens'] / metrics['count']:.1f}")
</syntaxhighlight>

=== Example 4: Streaming with Adapter Tracking ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True
)
engine = LLMEngine.from_engine_args(engine_args)

lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")
sampling_params = SamplingParams(max_tokens=128)

engine.add_request("stream-req", "Long SQL query", sampling_params, lora_request=lora_request)

# Stream partial outputs
print(f"Streaming with adapter: {lora_request.name}")
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        # Verify adapter consistency in partial outputs
        if output.lora_request:
            assert output.lora_request.name == lora_request.name

        # Display partial result
        if not output.finished:
            print(f"Partial: {output.outputs[0].text}", end="", flush=True)
        else:
            print(f"\nFinal: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 5: Multi-Sampling with LoRA (n>1) ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True
)
engine = LLMEngine.from_engine_args(engine_args)

lora_request = LoRARequest("creative-lora", 1, "/path/to/creative-lora")

# Generate multiple samples with same adapter
sampling_params = SamplingParams(
    n=5,  # Generate 5 different completions
    temperature=0.8,
    max_tokens=128
)

engine.add_request("multi-sample", "Write a story", sampling_params, lora_request=lora_request)

while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(f"Adapter: {output.lora_request.name}")
            print(f"Generated {len(output.outputs)} completions:\n")

            # All completions used same adapter
            for completion in output.outputs:
                assert completion.lora_request.name == lora_request.name
                print(f"Sample {completion.index + 1}:")
                print(f"  {completion.text}\n")
</syntaxhighlight>

=== Example 6: Adapter-Aware Error Handling ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True
)
engine = LLMEngine.from_engine_args(engine_args)

lora_request = LoRARequest("experimental-lora", 1, "/path/to/experimental-lora")
sampling_params = SamplingParams(max_tokens=128)

engine.add_request("test-req", "Test prompt", sampling_params, lora_request=lora_request)

try:
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.finished:
                # Check if generation failed
                if output.outputs[0].finish_reason == "error":
                    adapter_info = f"with adapter {output.lora_request.name}" if output.lora_request else "with base model"
                    print(f"Generation failed {adapter_info}")
                else:
                    print(f"Success: {output.outputs[0].text}")

except Exception as e:
    print(f"Error during inference: {e}")
    # Adapter information still available for debugging
    if hasattr(e, 'lora_request'):
        print(f"Failed adapter: {e.lora_request.name}")
</syntaxhighlight>

=== Example 7: Audit Trail with Adapter Info ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest
import json
from datetime import datetime

engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True
)
engine = LLMEngine.from_engine_args(engine_args)

# Audit log
audit_log = []

lora_request = LoRARequest("compliance-lora", 1, "/path/to/compliance-lora")
sampling_params = SamplingParams(max_tokens=128)

engine.add_request("audit-req-1", "Sensitive query", sampling_params, lora_request=lora_request)

while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            # Create audit record
            audit_record = {
                "timestamp": datetime.now().isoformat(),
                "request_id": output.request_id,
                "prompt": output.prompt,
                "response": output.outputs[0].text,
                "adapter_name": output.lora_request.name if output.lora_request else None,
                "adapter_id": output.lora_request.adapter_id if output.lora_request else None,
                "adapter_path": output.lora_request.path if output.lora_request else None,
                "finish_reason": output.outputs[0].finish_reason,
            }

            audit_log.append(audit_record)

# Save audit trail
with open("audit_log.json", "w") as f:
    json.dump(audit_log, f, indent=2)

print(f"Logged {len(audit_log)} inference operations with full adapter traceability")
</syntaxhighlight>

== Best Practices ==

* '''Always Check lora_request''': Use conditional checks since it may be None for base model outputs
* '''Preserve Metadata''': When forwarding outputs, maintain LoRARequest references for traceability
* '''Consistent Comparison''': Use adapter_id for equality checks, not name (names may not be unique)
* '''Metrics Aggregation''': Group metrics by adapter_id for accurate per-adapter analysis
* '''Logging''': Include adapter name and ID in all logs for debugging

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_LoRA_Output_Processing]]
* [[related_to::vllm-project_vllm_LoRARequest]]
* [[related_to::vllm-project_vllm_LLMEngine_step]]
