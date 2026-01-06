{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Attention]], [[domain::MLA]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks torch.cat vs direct copy optimization for k_nope/k_pe concatenation in Multi-head Latent Attention (MLA) prefill.

=== Description ===
This focused benchmark validates the optimization from commit 8d4142bd that replaces torch.cat with direct memory copy for concatenating k_nope and k_pe tensors in MLA attention mechanisms. The original approach used torch.cat with expand operations, which introduced unnecessary overhead. The optimized version pre-allocates the output tensor and directly copies the two input tensors, avoiding expand and concatenation overhead.

The benchmark tests across batch sizes from 32 to 65536 tokens using DeepSeek-V3 MLA dimensions (128 heads, 128 qk_nope_head_dim, 64 pe_dim). It measures performance for both bfloat16 and float8_e4m3fn dtypes, reporting speedup and latency reduction percentages. Results confirm that the direct copy method becomes beneficial at batch sizes >= 512, which is typical for prefill operations, with average speedups around 1.5-2x for large batches.

=== Usage ===
Run this benchmark when validating MLA attention optimizations, comparing tensor concatenation methods, or verifying performance improvements for DeepSeek-V3 and similar MLA-based models across different batch sizes.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_mla_k_concat.py benchmarks/kernels/benchmark_mla_k_concat.py]

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_method(
    method: Callable,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run benchmark (no command-line arguments)
python benchmarks/kernels/benchmark_mla_k_concat.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| k_nope || torch.Tensor || Key tensor without positional encoding [B, H, D_nope]
|-
| k_pe || torch.Tensor || Positional encoding key tensor [B, 1, D_pe]
|-
| num_warmup || int || Warmup iterations (default: 10)
|-
| num_iters || int || Benchmark iterations (default: 100)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_size || int || Number of tokens tested
|-
| cat_time || float || torch.cat latency in milliseconds
|-
| direct_time || float || Direct copy latency in milliseconds
|-
| speedup || float || Speedup ratio (cat_time / direct_time)
|-
| reduction || float || Latency reduction percentage
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Original torch.cat method
def cat_method(k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    return torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

# Optimized direct copy method
def direct_copy_method(k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    k = torch.empty(
        (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
        dtype=k_nope.dtype,
        device=k_nope.device
    )
    k[..., :k_nope.shape[-1]] = k_nope
    k[..., k_nope.shape[-1]:] = k_pe
    return k

# Benchmark usage
k_nope = torch.randn(batch_size, NUM_HEADS, QK_NOPE_HEAD_DIM,
                    dtype=dtype, device="cuda")
k_pe = torch.randn(batch_size, 1, PE_DIM,
                  dtype=dtype, device="cuda")

cat_time = benchmark_method(cat_method, k_nope, k_pe)
direct_time = benchmark_method(direct_copy_method, k_nope, k_pe)
speedup = cat_time / direct_time
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Concept:Multi_Head_Latent_Attention]]
* [[Concept:DeepSeek_V3]]
* [[Concept:Memory_Optimization]]
