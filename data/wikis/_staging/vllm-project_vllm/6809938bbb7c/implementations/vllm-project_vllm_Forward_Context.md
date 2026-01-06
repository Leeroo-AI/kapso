{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Execution_Context]], [[domain::Data_Parallelism]], [[domain::Model_Execution]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Forward pass context management system for tracking metadata, distributed state, and CUDA graph configuration during model execution.

=== Description ===
The forward_context.py module is a 358-line system that manages context information during neural network forward passes in vLLM. It provides a thread-safe mechanism to store and access metadata that needs to be available throughout the forward pass execution, including attention metadata, virtual engine indices, data parallel coordination data, and CUDA graph runtime modes.

The module defines key data structures: (1) BatchDescriptor - describes batch properties for CUDA graph dispatching (num_tokens, num_reqs, uniformity, LoRA status); (2) DPMetadata - manages data parallel state including token counts across DP ranks, max tokens, and chunked execution support with context managers for SP (sequence parallel) local sizes; (3) ForwardContext - the main context holding no-compile layers, attention metadata, virtual engine index, DP metadata, CUDA graph mode, batch descriptor, and micro-batch slicing information.

The module provides context managers (set_forward_context, override_forward_context) that ensure proper setup and teardown of forward pass state. It also includes optional batch size tracking and performance logging when VLLM_LOG_BATCHSIZE_INTERVAL is set. The context is globally accessible via get_forward_context() during execution, allowing layers deep in the model to access necessary metadata without explicit parameter passing.

=== Usage ===
Used internally by vLLM's execution engines to establish context before model forward passes. Accessed by model layers via get_forward_context().

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/forward_context.py vllm/forward_context.py]
* '''Lines:''' 1-358

=== Signature ===
<syntaxhighlight lang="python">
# Data structures
@dataclass
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False

    def relax_for_mixed_batch_cudagraphs(self) -> "BatchDescriptor"

@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    num_tokens_across_dp_cpu: torch.Tensor
    local_sizes: list[int] | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata"

    @contextmanager
    def chunked_sizes(
        self,
        sequence_parallel_size: int,
        max_chunk_size_per_rank: int,
        chunk_idx: int
    )

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int)

    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor

@dataclass
class ForwardContext:
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    virtual_engine: int
    dp_metadata: DPMetadata | None = None
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None
    ubatch_slices: UBatchSlices | None = None

# Context management
def get_forward_context() -> ForwardContext
def is_forward_context_available() -> bool
def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
) -> ForwardContext

@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.forward_context import (
    get_forward_context,
    set_forward_context,
    BatchDescriptor,
    DPMetadata,
)
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| BatchDescriptor || NamedTuple || Describes batch properties for CUDA graph dispatch
|-
| DPMetadata || Dataclass || Data parallel metadata and coordination
|-
| ForwardContext || Dataclass || Complete forward pass context information
|-
| get_forward_context || Function || Retrieve current forward context
|-
| is_forward_context_available || Function || Check if context is set
|-
| create_forward_context || Function || Create a new forward context
|-
| set_forward_context || ContextManager || Set context for forward pass duration
|-
| override_forward_context || ContextManager || Temporarily override current context
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import torch
from vllm.forward_context import (
    set_forward_context,
    get_forward_context,
    BatchDescriptor,
    DPMetadata,
)
from vllm.config import VllmConfig, CUDAGraphMode

# Example 1: Basic forward context setup
vllm_config = VllmConfig(...)
attn_metadata = prepare_attention_metadata(...)

with set_forward_context(
    attn_metadata=attn_metadata,
    vllm_config=vllm_config,
    virtual_engine=0,
    num_tokens=1024,
):
    # Forward pass happens here
    output = model.forward(input_ids)
    # Context is automatically cleaned up

# Example 2: Accessing context in model layers
class MyAttentionLayer(torch.nn.Module):
    def forward(self, hidden_states):
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        # Use attention metadata for this layer
        return self.attention(hidden_states, attn_metadata)

# Example 3: Data parallel execution
if vllm_config.parallel_config.data_parallel_size > 1:
    num_tokens_across_dp = coordinate_batch_across_dp(...)

    with set_forward_context(
        attn_metadata=attn_metadata,
        vllm_config=vllm_config,
        num_tokens=local_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        ctx = get_forward_context()
        dp_metadata = ctx.dp_metadata
        print(f"Max tokens across DP: {dp_metadata.max_tokens_across_dp_cpu}")

# Example 4: CUDA graph execution
batch_desc = BatchDescriptor(
    num_tokens=256,
    num_reqs=8,
    uniform=True,
    has_lora=False,
)

with set_forward_context(
    attn_metadata=attn_metadata,
    vllm_config=vllm_config,
    num_tokens=256,
    cudagraph_runtime_mode=CUDAGraphMode.FULL,
    batch_descriptor=batch_desc,
):
    # This forward pass can use CUDA graphs
    output = model.forward(input_ids)

# Example 5: Chunked execution for large batches
ctx = get_forward_context()
dp_meta = ctx.dp_metadata

# Process in chunks
for chunk_idx in range(num_chunks):
    with dp_meta.chunked_sizes(
        sequence_parallel_size=tp_size,
        max_chunk_size_per_rank=max_chunk,
        chunk_idx=chunk_idx,
    ) as local_sizes:
        # Process chunk with proper DP coordination
        chunk_output = process_chunk(chunk_idx, local_sizes)

# Example 6: Batch descriptor for CUDA graph dispatch
def get_cuda_graph_key(num_tokens, num_reqs):
    desc = BatchDescriptor(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        uniform=True,
    )
    # Use descriptor as CUDA graph key
    return desc
</syntaxhighlight>

== Related Pages ==
* [[uses::Module:vllm-project_vllm_Attention_Metadata]]
* [[implements::Pattern:Context_Manager]]
* [[related::Module:vllm-project_vllm_Parallel_Config]]
* [[used_by::Module:vllm-project_vllm_Model_Runner]]
