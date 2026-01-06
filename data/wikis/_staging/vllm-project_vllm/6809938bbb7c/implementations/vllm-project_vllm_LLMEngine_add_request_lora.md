{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::LoRA]], [[domain::Request_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for submitting inference requests with a specific LoRA adapter to the vLLM engine.

=== Description ===

The `add_request` method (or `generate()` for the LLM class) accepts a `lora_request` parameter that specifies which LoRA adapter to use for that specific request. This enables:
- Per-request adapter selection
- Mixing different adapters in the same batch
- Dynamic adapter routing based on user or task

The engine handles adapter loading, caching, and application automatically.

=== Usage ===

Submit LoRA requests when:
- Running inference with a specific fine-tuned adapter
- Implementing per-user or per-task adapter routing
- Batching requests that use different adapters
- A/B testing different adapter versions

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/v1/engine/llm_engine.py (core), vllm/entrypoints/llm.py (LLM class)
* '''Lines:''' L100-200 (llm_engine.py), L365-434 (llm.py generate method)

=== Signature ===
<syntaxhighlight lang="python">
# Via LLM class (recommended)
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | None = None,
    *,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    # ... other parameters
) -> list[RequestOutput]:
    """
    Generate with optional LoRA adapter.

    Args:
        prompts: Input prompts.
        sampling_params: Generation parameters.
        lora_request: LoRA adapter(s) to use.
            - Single LoRARequest: Apply to all prompts
            - List of LoRARequest: One per prompt

    Returns:
        List of RequestOutput with generated text.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| prompts || PromptType | Sequence[PromptType] || Yes || Input prompts
|-
| sampling_params || SamplingParams || No || Generation parameters
|-
| lora_request || LoRARequest | list[LoRARequest] || No || Adapter(s) to use
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs || list[RequestOutput] || Generated text with adapter applied
|}

== Usage Examples ==

=== Single Adapter for All Prompts ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=2,
)

# Define adapter
sql_lora = LoRARequest(
    lora_name="sql",
    lora_int_id=1,
    lora_path="yard1/llama-2-7b-sql-lora-test",
)

# Apply same adapter to all prompts
prompts = [
    "Convert to SQL: Get all users older than 30",
    "Convert to SQL: Count products by category",
]

outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=100),
    lora_request=sql_lora,  # Applied to all
)

for output in outputs:
    print(output.outputs[0].text)
</syntaxhighlight>

=== Different Adapters Per Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=4,
)

# Define multiple adapters
sql_lora = LoRARequest("sql", 1, "yard1/llama-2-7b-sql-lora-test")
code_lora = LoRARequest("code", 2, "my-org/llama-code-lora")

# Different adapters for different prompts
prompts = [
    "Convert to SQL: Get all users",     # Use SQL adapter
    "Write Python to sort a list",        # Use code adapter
]

outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=100),
    lora_request=[sql_lora, code_lora],  # Per-prompt
)
</syntaxhighlight>

=== Dynamic Adapter Selection ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=4,
)

# Adapter registry
adapters = {
    "sql": LoRARequest("sql", 1, "org/sql-lora"),
    "code": LoRARequest("code", 2, "org/code-lora"),
    "math": LoRARequest("math", 3, "org/math-lora"),
}

def route_request(task_type, prompt):
    """Route request to appropriate adapter."""
    adapter = adapters.get(task_type)
    return llm.generate(
        [prompt],
        SamplingParams(max_tokens=200),
        lora_request=adapter,
    )[0]

# Use different adapters based on task
result1 = route_request("sql", "Convert: Show all orders")
result2 = route_request("code", "Write: Fibonacci function")
result3 = route_request("math", "Solve: 2x + 5 = 15")
</syntaxhighlight>

=== Mixed Batch (Some with LoRA, Some Without) ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=2,
)

sql_lora = LoRARequest("sql", 1, "yard1/llama-2-7b-sql-lora-test")

# Mix of base model and adapter requests
prompts = [
    "What is the capital of France?",  # Base model (no adapter)
    "Convert to SQL: Get all users",   # With adapter
]

# Use list with None for base model
outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=100),
    lora_request=[None, sql_lora],
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_LoRA_Request_Submission]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
