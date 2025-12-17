'''Metadata'''
{| class="wikitable"
|-
! Key !! Value
|-
| Knowledge Sources || vLLM Source Code, LoRA Request API
|-
| Domains || Request Metadata, Adapter Routing
|-
| Last Updated || 2025-12-17
|}

== Overview ==

'''LoRARequest''' is the data structure used to associate inference requests with specific LoRA adapters in vLLM. This class encapsulates adapter identification, path information, and optional configuration parameters required for multi-LoRA serving.

== Description ==

The LoRARequest class (implemented using msgspec.Struct for performance) serves as the container for all adapter-related metadata attached to generation requests. When a client submits a prompt for inference with a specific LoRA adapter, they create a LoRARequest instance and pass it to the engine's add_request or generate methods.

=== Class Structure ===

LoRARequest is defined as a structured data class with the following fields:

* '''lora_name''' (str): Human-readable identifier for the adapter (e.g., "sql-lora", "math-adapter")
* '''lora_int_id''' (int): Unique integer identifier >= 1 used for internal routing
* '''lora_path''' (str): Path to adapter weights (local filesystem, HF repo ID, or URL)
* '''lora_local_path''' (str | None): Deprecated field, use lora_path instead
* '''long_lora_max_len''' (int | None): Maximum sequence length for LongLoRA adapters
* '''base_model_name''' (str | None): Base model identifier for validation
* '''tensorizer_config_dict''' (dict | None): Configuration for tensorized adapter loading

The class implements custom __eq__ and __hash__ methods based on lora_name, allowing LoRARequest instances to be used as dictionary keys and set members. This enables efficient adapter deduplication and cache lookups.

=== Validation ===

The __post_init__ method enforces:
* lora_int_id must be >= 1 (0 reserved for base model)
* lora_path cannot be empty
* Deprecation warnings for lora_local_path usage

=== Properties ===

Convenience properties provide consistent access:
* adapter_id: Alias for lora_int_id
* name: Alias for lora_name
* path: Alias for lora_path

== Code Reference ==

=== Source Location ===
* '''File''': vllm/lora/request.py
* '''Class''': LoRARequest

=== Signature ===

<syntaxhighlight lang="python">
class LoRARequest(msgspec.Struct, omit_defaults=True, array_like=True):
    """
    Request for a LoRA adapter.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    lora_local_path: str | None = None
    long_lora_max_len: int | None = None
    base_model_name: str | None = None
    tensorizer_config_dict: dict | None = None

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")
        assert self.lora_path, "lora_path cannot be empty"

    @property
    def adapter_id(self) -> int:
        return self.lora_int_id
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest
</syntaxhighlight>

== I/O Contract ==

=== Input Parameters ===

{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| lora_name || str || required || Human-readable adapter name (for logging/monitoring)
|-
| lora_int_id || int || required || Unique ID >= 1 for adapter routing
|-
| lora_path || str || "" || Path to adapter weights (local path or HF repo ID)
|-
| lora_local_path || str \| None || None || Deprecated - use lora_path
|-
| long_lora_max_len || int \| None || None || Max sequence length for LongLoRA adapters
|-
| base_model_name || str \| None || None || Base model identifier for validation
|-
| tensorizer_config_dict || dict \| None || None || Tensorizer configuration
|}

=== Output ===

{| class="wikitable"
|-
! Return Type !! Description
|-
| LoRARequest || Immutable request object ready for use with engine methods
|}

== Usage Examples ==

=== Example 1: Basic LoRA Request ===

<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Create request for SQL adapter
lora_request = LoRARequest(
    lora_name="sql-lora",
    lora_int_id=1,
    lora_path="/path/to/sql-lora"
)

print(f"Adapter: {lora_request.name}")
print(f"ID: {lora_request.adapter_id}")
print(f"Path: {lora_request.path}")
</syntaxhighlight>

=== Example 2: Hugging Face Hub Adapter ===

<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Use HF repository ID directly
lora_request = LoRARequest(
    lora_name="text2sql",
    lora_int_id=1,
    lora_path="jeeejeee/llama32-3b-text2sql-spider"
)

# Path will be resolved by engine during loading
</syntaxhighlight>

=== Example 3: Multiple Adapters ===

<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Create requests for different adapters
math_lora = LoRARequest("math-lora", 1, "/path/to/math-lora")
code_lora = LoRARequest("code-lora", 2, "/path/to/code-lora")
sql_lora = LoRARequest("sql-lora", 3, "/path/to/sql-lora")

# Each has unique ID for routing
adapters = {
    "math": math_lora,
    "code": code_lora,
    "sql": sql_lora
}
</syntaxhighlight>

=== Example 4: Using with LLM.generate() ===

<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Initialize LoRA-enabled LLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_lora_rank=64
)

# Create adapter request
lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")

# Generate with adapter
prompts = ["Write a SQL query to get all users"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

for output in outputs:
    print(f"Result: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 5: Using with LLMEngine.add_request() ===

<syntaxhighlight lang="python">
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

# Initialize engine
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_loras=2
)
engine = LLMEngine.from_engine_args(engine_args)

# Create LoRA request
lora_request = LoRARequest("sql-lora", 1, "/path/to/sql-lora")

# Add request to engine
request_id = "request-001"
prompt = "Generate SQL query"
sampling_params = SamplingParams(max_tokens=128)

engine.add_request(
    request_id=request_id,
    prompt=prompt,
    sampling_params=sampling_params,
    lora_request=lora_request
)

# Process requests
while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(f"Output: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 6: Adapter Request with Downloaded Weights ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest

# Download adapter from HF Hub
lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")

# Create request with local path
lora_request = LoRARequest(
    lora_name="text2sql",
    lora_int_id=1,
    lora_path=lora_path
)

print(f"Adapter loaded from: {lora_request.path}")
</syntaxhighlight>

=== Example 7: Dynamic Adapter Selection ===

<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Adapter registry
ADAPTERS = {
    "math": LoRARequest("math-lora", 1, "/path/to/math-lora"),
    "code": LoRARequest("code-lora", 2, "/path/to/code-lora"),
    "sql": LoRARequest("sql-lora", 3, "/path/to/sql-lora"),
}

# Select adapter based on task
def get_lora_for_task(task: str) -> LoRARequest | None:
    return ADAPTERS.get(task)

# Use in request routing
task = "sql"
lora_request = get_lora_for_task(task)

if lora_request:
    print(f"Using adapter: {lora_request.name}")
else:
    print("Using base model")
</syntaxhighlight>

=== Example 8: Validation and Error Handling ===

<syntaxhighlight lang="python">
from vllm.lora.request import LoRARequest

# Invalid ID (must be >= 1)
try:
    invalid_request = LoRARequest("test", 0, "/path/to/lora")
except ValueError as e:
    print(f"Error: {e}")  # "id must be > 0, got 0"

# Empty path
try:
    invalid_request = LoRARequest("test", 1, "")
except AssertionError as e:
    print(f"Error: lora_path cannot be empty")

# Valid request
valid_request = LoRARequest("valid-lora", 1, "/path/to/lora")
print(f"Created valid request: {valid_request.name}")
</syntaxhighlight>

== Best Practices ==

* '''ID Uniqueness''': Use a centralized ID registry to prevent collisions in production
* '''Naming Convention''': Use descriptive names that indicate adapter purpose (e.g., "llama-sql-v2")
* '''Path Management''': Use absolute paths or HF repo IDs for consistency
* '''Request Reuse''': Create LoRARequest objects once and reuse across multiple prompts
* '''Documentation''': Maintain a mapping of adapter IDs to their purposes for operations

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_LoRA_Request_Creation]]
* [[related_to::vllm-project_vllm_LLMEngine_add_request]]
* [[related_to::vllm-project_vllm_LLM_generate]]
