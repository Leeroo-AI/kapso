{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::PyTorch_Internals]], [[domain::Monkey_Patching]], [[domain::Compilation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
PyTorch monkey patches and environment configurations that are applied on vLLM import to work around known issues.

=== Description ===
The env_override.py module is a 378-line collection of critical monkey patches and environment overrides that are automatically applied when vLLM is imported. These modifications address specific bugs and incompatibilities in PyTorch 2.9, particularly related to torch.compile and TorchInductor's graph partitioning and memory planning.

The module contains three major monkeypatches: (1) Memory plan reuse patch for PythonWrapperCodegen that fixes a test failure in multi-graph piecewise compilation by properly handling output names in subgraph contexts; (2) Graph partition signature patch that fixes inductor partition + attention-nvfp4 quant fusion by correctly handling NoneLayout buffers and WeakDep dependencies; (3) Scheduler should_partition patch that works around a bug where operators with in-place mutations inside splitting_ops don't populate the origin_node field, causing assertion errors when use_inductor_graph_partition is enabled.

The module also sets important environment variables: PYTORCH_NVML_BASED_CUDA_CHECK=1 to avoid unintentional CUDA initialization, and TORCHINDUCTOR_COMPILE_THREADS=1 to prevent threading issues. These patches are version-specific (only applied to PyTorch 2.9.0) and include detailed comments linking to upstream PyTorch PRs and vLLM issues.

=== Usage ===
Automatically executed when vLLM is imported. Should not be called directly by user code.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/env_override.py vllm/env_override.py]
* '''Lines:''' 1-378

=== Signature ===
<syntaxhighlight lang="python">
# Environment variable overrides
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
torch._inductor.config.compile_threads = 1

# Monkeypatch functions
def memory_plan_reuse_patched(self)
def get_graph_partition_signature_patched(
    self,
    partitions,
    skip_cudagraphs: list[bool]
)
def should_partition_patched(
    self,
    node,
    should_log: bool = False
) -> bool
def _update_scheduler_patched(self) -> None

# Conditional application for PyTorch 2.9.0
if is_torch_equal("2.9.0"):
    # Apply patches to PyTorch internals
    PythonWrapperCodegen.memory_plan_reuse = memory_plan_reuse_patched
    GraphLowering._update_scheduler = _update_scheduler_patched
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This module is automatically imported by vLLM
# No direct import needed - patches are applied on module import
import vllm  # Patches are applied here
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| memory_plan_reuse_patched || Function || Patched memory planning for subgraphs
|-
| get_graph_partition_signature_patched || Function || Fixed graph partition signature computation
|-
| should_partition_patched || Function || Patched partition decision logic
|-
| _update_scheduler_patched || Function || Scheduler initialization with patches applied
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Patches are applied automatically
import vllm
# At this point, all PyTorch 2.9.0 patches have been applied

# Example 2: Verify environment overrides
import os
print(os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"))  # "1"
print(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"))  # "1"

# Example 3: Check if patches are active
import torch
from vllm.utils.torch_utils import is_torch_equal

if is_torch_equal("2.9.0"):
    print("PyTorch 2.9.0 detected - vLLM patches are active")
    # The following have been monkeypatched:
    # - torch._inductor.codegen.wrapper.PythonWrapperCodegen.memory_plan_reuse
    # - torch._inductor.graph.GraphLowering._update_scheduler
    # - torch._inductor.scheduler.Scheduler.should_partition

# Example 4: Understanding patch purpose
# These patches fix specific bugs:

# Bug 1: Multi-graph piecewise compilation
# See: https://github.com/pytorch/pytorch/pull/165514
# Fixed by memory_plan_reuse_patched

# Bug 2: Inductor partition + attention-nvfp4 quant fusion
# See: https://github.com/pytorch/pytorch/pull/165815
# Fixed by get_graph_partition_signature_patched

# Bug 3: In-place mutations in splitting_ops
# See: https://github.com/vllm-project/vllm/issues/26678
# Fixed by should_partition_patched

# Example 5: Impact on vLLM compilation
# These patches enable vLLM to use torch.compile with:
# - Graph partitioning for CUDA graphs
# - Custom partitioning for vLLM operators
# - Better memory planning for subgraphs
# - Support for attention quantization fusion

from vllm import LLM

# Compilation works correctly with patches
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enforce_eager=False,  # torch.compile enabled
    # Patches ensure compilation works correctly
)
</syntaxhighlight>

== Related Pages ==
* [[patches::Framework:PyTorch]]
* [[fixes::Bug:PyTorch_2.9_Inductor]]
* [[requires::Module:vllm-project_vllm_Torch_Utils]]
* [[related::Module:vllm-project_vllm_Environment_Variables]]
