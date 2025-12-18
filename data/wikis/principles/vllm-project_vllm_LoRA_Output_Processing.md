{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::NLP]], [[domain::LoRA]], [[domain::Output_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of extracting, verifying, and utilizing outputs from LoRA-enabled inference including adapter attribution and quality analysis.

=== Description ===

LoRA Output Processing extends standard output handling with adapter-specific information:

1. **Adapter Attribution:** Which adapter generated this output
2. **Verification:** Confirming the intended adapter was applied
3. **Audit Logging:** Recording adapter usage for compliance
4. **Quality Analysis:** Comparing outputs across different adapters
5. **Error Handling:** Detecting adapter application failures

This information is essential for multi-tenant systems and debugging adapter behavior.

=== Usage ===

Process LoRA outputs when:
- Building auditable multi-tenant LLM systems
- Debugging adapter routing issues
- Analyzing adapter effectiveness
- Implementing quality assurance for fine-tuned models
- Creating A/B testing frameworks for adapters

== Theoretical Basis ==

'''Output Attribution Model:'''

Each output carries its adapter lineage:

<syntaxhighlight lang="python">
# Output structure with adapter info
output = {
    "request_id": "req-123",
    "prompt": "...",
    "generated_text": "...",
    "adapter": {
        "name": "sql-adapter",
        "id": 1,
        "path": "yard1/llama-2-7b-sql-lora-test",
    }
}
</syntaxhighlight>

'''Verification Pattern:'''

<syntaxhighlight lang="python">
def verify_adapter_application(output, expected_adapter):
    """
    Verify the correct adapter was applied.

    Args:
        output: RequestOutput from generation
        expected_adapter: LoRARequest that should have been used

    Raises:
        AssertionError if adapter mismatch
    """
    actual = output.lora_request

    if expected_adapter is None:
        assert actual is None, f"Expected base model, got {actual.lora_name}"
    else:
        assert actual is not None, "Expected adapter, got base model"
        assert actual.lora_name == expected_adapter.lora_name, \
            f"Adapter mismatch: {expected_adapter.lora_name} != {actual.lora_name}"
</syntaxhighlight>

'''Audit Logging Pattern:'''

<syntaxhighlight lang="python">
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AdapterAuditRecord:
    timestamp: datetime
    request_id: str
    adapter_name: str | None
    adapter_path: str | None
    prompt_hash: str
    output_tokens: int
    latency_ms: float

def create_audit_record(output, latency_ms):
    """Create audit record for LoRA output."""
    adapter = output.lora_request

    return AdapterAuditRecord(
        timestamp=datetime.now(),
        request_id=output.request_id,
        adapter_name=adapter.lora_name if adapter else None,
        adapter_path=adapter.lora_path if adapter else None,
        prompt_hash=hash(output.prompt),
        output_tokens=len(output.outputs[0].token_ids),
        latency_ms=latency_ms,
    )
</syntaxhighlight>

'''Quality Analysis Pattern:'''

<syntaxhighlight lang="python">
def compare_adapter_outputs(llm, prompt, adapters, sampling_params):
    """
    Compare outputs across different adapters.

    Returns dict of adapter_name -> output for analysis.
    """
    results = {}

    for name, adapter in adapters.items():
        outputs = llm.generate(
            [prompt],
            sampling_params,
            lora_request=adapter,
        )
        results[name] = {
            "text": outputs[0].outputs[0].text,
            "finish_reason": outputs[0].outputs[0].finish_reason,
            "tokens": len(outputs[0].outputs[0].token_ids),
        }

    return results
</syntaxhighlight>

'''Error Detection:'''

<syntaxhighlight lang="python">
def check_adapter_errors(outputs, expected_adapters):
    """Check for adapter application errors."""
    errors = []

    for output, expected in zip(outputs, expected_adapters):
        actual = output.lora_request

        if expected is None and actual is not None:
            errors.append({
                "request_id": output.request_id,
                "error": "unexpected_adapter",
                "expected": None,
                "actual": actual.lora_name,
            })
        elif expected is not None and actual is None:
            errors.append({
                "request_id": output.request_id,
                "error": "missing_adapter",
                "expected": expected.lora_name,
                "actual": None,
            })
        elif expected and actual and expected.lora_name != actual.lora_name:
            errors.append({
                "request_id": output.request_id,
                "error": "adapter_mismatch",
                "expected": expected.lora_name,
                "actual": actual.lora_name,
            })

    return errors
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_RequestOutput_lora]]
