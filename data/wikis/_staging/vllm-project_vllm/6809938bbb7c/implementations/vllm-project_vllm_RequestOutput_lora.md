{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::LoRA]], [[domain::Output_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for extracting generated text and adapter attribution from LoRA-enabled inference results.

=== Description ===

`RequestOutput` in LoRA-enabled inference includes the `lora_request` field that indicates which adapter was used to generate the output. This enables:
- Verification that the correct adapter was applied
- Logging and auditing of adapter usage
- Debugging multi-adapter request routing
- Associating outputs with their adapter sources

=== Usage ===

Access LoRA output fields when:
- Verifying adapter application in multi-tenant systems
- Building audit trails for adapter usage
- Debugging adapter routing issues
- Analyzing output quality by adapter

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/outputs.py
* '''Lines:''' L23-63 (CompletionOutput), L84-191 (RequestOutput)

=== Interface ===
<syntaxhighlight lang="python">
class RequestOutput:
    """Output from a LoRA-enabled generation request."""

    request_id: str
    """Unique request identifier."""

    prompt: str | None
    """Original input prompt."""

    outputs: list[CompletionOutput]
    """Generated completions."""

    lora_request: LoRARequest | None
    """LoRA adapter used for this request (if any)."""

    # ... other fields


class CompletionOutput:
    """Single completion output."""

    text: str
    """Generated text."""

    token_ids: Sequence[int]
    """Generated token IDs."""

    lora_request: LoRARequest | None
    """LoRA adapter used (for per-completion tracking)."""

    # ... other fields
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import RequestOutput
from vllm.outputs import CompletionOutput
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| RequestOutput || RequestOutput || Yes || Output from LLM.generate()
|}

=== Outputs (Accessible Fields) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs[0].text || str || Generated text
|-
| lora_request || LoRARequest | None || Adapter used (None = base model)
|-
| lora_request.lora_name || str || Adapter name (if adapter used)
|}

== Usage Examples ==

=== Basic Adapter Attribution ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
)

adapter = LoRARequest("sql", 1, "yard1/llama-2-7b-sql-lora-test")

outputs = llm.generate(
    ["Convert to SQL: Get all users"],
    SamplingParams(max_tokens=100),
    lora_request=adapter,
)

output = outputs[0]

# Check which adapter was used
if output.lora_request:
    print(f"Generated with adapter: {output.lora_request.lora_name}")
    print(f"Adapter path: {output.lora_request.lora_path}")
else:
    print("Generated with base model")

print(f"Output: {output.outputs[0].text}")
</syntaxhighlight>

=== Audit Logging for Multi-Adapter Requests ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
from datetime import datetime

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=4,
)

adapters = {
    "sql": LoRARequest("sql", 1, "org/sql-lora"),
    "code": LoRARequest("code", 2, "org/code-lora"),
}

def generate_with_audit(prompt, adapter_name):
    """Generate with audit logging."""
    adapter = adapters.get(adapter_name)

    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=100),
        lora_request=adapter,
    )

    output = outputs[0]

    # Create audit record
    audit_record = {
        "timestamp": datetime.now().isoformat(),
        "request_id": output.request_id,
        "prompt": prompt[:100],  # Truncate for logging
        "adapter_requested": adapter_name,
        "adapter_applied": output.lora_request.lora_name if output.lora_request else None,
        "output_length": len(output.outputs[0].text),
    }

    print(json.dumps(audit_record))
    return output

# Usage
generate_with_audit("Convert: SELECT * FROM users", "sql")
generate_with_audit("Write Python: Hello World", "code")
</syntaxhighlight>

=== Verifying Adapter Application ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=2,
)

sql_adapter = LoRARequest("sql", 1, "yard1/llama-2-7b-sql-lora-test")

outputs = llm.generate(
    ["Convert to SQL: Get users"],
    SamplingParams(max_tokens=50),
    lora_request=sql_adapter,
)

output = outputs[0]

# Verify correct adapter was applied
assert output.lora_request is not None, "Expected adapter to be applied"
assert output.lora_request.lora_name == "sql", f"Wrong adapter: {output.lora_request.lora_name}"

print("✓ Correct adapter applied")
print(f"Output: {output.outputs[0].text}")
</syntaxhighlight>

=== Processing Mixed Adapter Batch ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=2,
)

sql_adapter = LoRARequest("sql", 1, "org/sql-lora")
code_adapter = LoRARequest("code", 2, "org/code-lora")

# Mixed batch with different adapters
prompts = ["SQL query", "Python code", "Another SQL"]
adapters = [sql_adapter, code_adapter, sql_adapter]

outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=50),
    lora_request=adapters,
)

# Process outputs, grouping by adapter
results_by_adapter = {}
for output in outputs:
    adapter_name = output.lora_request.lora_name if output.lora_request else "base"

    if adapter_name not in results_by_adapter:
        results_by_adapter[adapter_name] = []

    results_by_adapter[adapter_name].append({
        "prompt": output.prompt[:30],
        "output": output.outputs[0].text[:50],
    })

# Print results grouped by adapter
for adapter, results in results_by_adapter.items():
    print(f"\n=== {adapter} adapter ===")
    for r in results:
        print(f"  {r['prompt']}... -> {r['output']}...")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_LoRA_Output_Processing]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
