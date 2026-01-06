{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::LoRA]], [[domain::Model_Serving]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for creating a LoRA adapter request object that identifies which adapter to use for a specific inference request.

=== Description ===

`LoRARequest` is a data class that encapsulates all information needed to identify and load a LoRA adapter for inference:
- **lora_name:** Human-readable identifier for the adapter
- **lora_int_id:** Unique integer ID (must be > 0)
- **lora_path:** Path to adapter weights (HuggingFace Hub or local)

Each request can specify a different adapter, enabling multi-tenant serving on a single base model.

=== Usage ===

Create `LoRARequest` objects when:
- Specifying which adapter to use for a generation request
- Switching between different fine-tuned models dynamically
- Building multi-user systems with personalized adapters
- A/B testing different adapter versions

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/lora/request.py
* '''Lines:''' L9-96

=== Signature ===
<syntaxhighlight lang="python">
class LoRARequest(msgspec.Struct):
    """Request for a LoRA adapter."""

    lora_name: str
    """Human-readable name for the adapter."""

    lora_int_id: int
    """Unique integer ID (must be > 0)."""

    lora_path: str = ""
    """Path to adapter weights (HuggingFace Hub ID or local path)."""

    long_lora_max_len: int | None = None
    """Maximum sequence length for long-context LoRA (S2 attention)."""

    base_model_name: str | None = None
    """Base model name (for validation)."""

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")
        assert self.lora_path, "lora_path cannot be empty"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| lora_name || str || Yes || Human-readable adapter identifier
|-
| lora_int_id || int || Yes || Unique integer ID (must be > 0)
|-
| lora_path || str || Yes || Path to adapter weights
|-
| long_lora_max_len || int || No || Max length for long-context LoRA
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| LoRARequest || LoRARequest || Configured adapter request object
|}

== Usage Examples ==

=== Basic LoRA Request ===
<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Create request for a HuggingFace adapter
adapter = LoRARequest(
    lora_name="sql-adapter",
    lora_int_id=1,
    lora_path="yard1/llama-2-7b-sql-lora-test",
)
</syntaxhighlight>

=== Local Adapter Path ===
<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Local adapter weights
adapter = LoRARequest(
    lora_name="custom-adapter",
    lora_int_id=2,
    lora_path="/data/lora_adapters/my_adapter",
)
</syntaxhighlight>

=== Multiple Adapters ===
<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Define multiple adapters (each needs unique int_id)
adapters = {
    "sql": LoRARequest(
        lora_name="sql-adapter",
        lora_int_id=1,
        lora_path="yard1/llama-2-7b-sql-lora-test",
    ),
    "code": LoRARequest(
        lora_name="code-adapter",
        lora_int_id=2,
        lora_path="my-org/llama-code-lora",
    ),
    "chat": LoRARequest(
        lora_name="chat-adapter",
        lora_int_id=3,
        lora_path="/local/chat_lora",
    ),
}
</syntaxhighlight>

=== Long-Context LoRA ===
<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Adapter with extended context support (S2 attention)
long_adapter = LoRARequest(
    lora_name="long-context-adapter",
    lora_int_id=4,
    lora_path="my-org/llama-32k-lora",
    long_lora_max_len=32768,  # Extended context
)
</syntaxhighlight>

=== Use with LLM.generate() ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Initialize with LoRA support
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_lora=True,
    max_loras=2,
)

# Create adapter request
sql_adapter = LoRARequest(
    lora_name="sql",
    lora_int_id=1,
    lora_path="yard1/llama-2-7b-sql-lora-test",
)

# Generate with adapter
outputs = llm.generate(
    ["Convert to SQL: Show all users"],
    SamplingParams(max_tokens=100),
    lora_request=sql_adapter,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_LoRA_Adapter_Registration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
