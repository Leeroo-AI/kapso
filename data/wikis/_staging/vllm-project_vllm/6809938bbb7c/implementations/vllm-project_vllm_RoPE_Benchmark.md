{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Kernel Benchmarking]], [[domain::Positional Encoding]], [[domain::Attention Mechanisms]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A benchmark comparing three RoPE (Rotary Position Embedding) implementations (PyTorch native, FlashInfer, vLLM) for transformer models.

=== Description ===
This benchmark evaluates Rotary Position Embedding (RoPE) kernel performance across three implementations: PyTorch native, FlashInfer's CUDA kernels, and vLLM's custom implementation. RoPE encodes positional information into query and key tensors for attention mechanisms. The benchmark tests performance across various batch sizes, sequence lengths, and number of attention heads, supporting both NeoX-style and standard RoPE variants. It measures execution time in microseconds using Triton's benchmarking framework.

=== Usage ===
Use this benchmark to select the optimal RoPE implementation for your hardware and model configuration, or to validate custom RoPE kernel implementations against reference versions.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_rope.py benchmarks/kernels/benchmark_rope.py]

=== Signature ===
<syntaxhighlight lang="python">
def get_benchmark(head_size: int, rotary_dim: int,
                  is_neox_style: bool, device: str) -> Callable
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python benchmarks/kernels/benchmark_rope.py \
    --head-size 128 --rotary-dim 32 --is-neox-style True \
    --device cuda:0 --save-path ./configs/rope/
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| positions || torch.Tensor || Position indices, shape (batch_size, seq_len)
|-
| query || torch.Tensor || Query tensor to apply RoPE encoding
|-
| key || torch.Tensor || Key tensor to apply RoPE encoding
|-
| --head-size || int || Attention head dimension (default 128)
|-
| --rotary-dim || int || Rotary dimension, typically 16 or 32
|-
| --is-neox-style || bool || Use NeoX-style RoPE variant (default True)
|-
| --batch-size || int || Batch size for testing (default 16)
|-
| --seq-len || int || Sequence length (default 512)
|-
| --num-heads || int || Number of attention heads (default 8)
|-
| --device || str || CUDA device to use (default cuda:0)
|-
| --save-path || str || Path to save benchmark results
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| query_encoded || torch.Tensor || Query with RoPE positional encoding applied
|-
| key_encoded || torch.Tensor || Key with RoPE positional encoding applied
|-
| benchmark_plots || File || Performance comparison plots
|-
| timing_data || CSV || Execution time metrics per implementation
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Run benchmark for standard configuration
python benchmarks/kernels/benchmark_rope.py \
    --head-size 128 \
    --rotary-dim 32 \
    --is-neox-style True \
    --device cuda:0 \
    --save-path ./rope_results/

# Use with custom model configuration
from vllm.model_executor.layers.rotary_embedding import get_rope
import torch

head_size = 128
max_position = 8192
is_neox_style = True
rope_parameters = {"partial_rotary_factor": 0.25}

rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
rope = rope.to(dtype=torch.bfloat16, device="cuda")

positions = torch.randint(0, max_position, (4, 512), device="cuda")
query = torch.randn((4, 512, 8 * 128), dtype=torch.bfloat16, device="cuda")
key = torch.randn_like(query)

# Apply RoPE encoding
rope.forward_cuda(positions, query, key)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:vllm_attention_layers]]
* [[Concept:Rotary_Position_Embedding]]
* [[Benchmark:Attention_Kernel_Performance]]
