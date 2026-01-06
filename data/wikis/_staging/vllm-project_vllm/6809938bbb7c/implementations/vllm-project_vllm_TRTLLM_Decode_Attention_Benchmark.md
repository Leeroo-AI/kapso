{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Kernel Benchmarking]], [[domain::Attention Mechanisms]], [[domain::Quantization]], [[domain::Decode Phase]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A benchmark comparing TensorRT-LLM decode attention kernels against FlashInfer baseline across various quantization configurations.

=== Description ===
This benchmark evaluates TensorRT-LLM's decode attention implementation (for token generation phase) with support for mixed-precision quantization of queries, KV cache, and outputs. It tests configurations including FP8 quantization for Q/K/V and FP4 output quantization, measuring performance improvements over the FlashInfer baseline. The benchmark uses paged KV cache and tests across varying batch sizes (1-256) and sequence lengths (1K-131K tokens). Results are saved to CSV for analysis.

=== Usage ===
Use this benchmark to evaluate TRT-LLM decode attention performance with different quantization strategies, compare against FlashInfer baseline for hardware-specific optimizations, or validate quantized attention correctness.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_trtllm_decode_attention.py benchmarks/kernels/benchmark_trtllm_decode_attention.py]

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_decode(
    dtype: torch.dtype,
    quant_dtypes: tuple[torch.dtype | None, torch.dtype | None, torch.dtype | None],
    batch_size: int,
    max_seq_len: int,
    num_heads: tuple[int, int] = (64, 8),
    head_size: int = 128,
    kv_layout: str = "HND",
    block_size: int = 16,
    warmup: int = 10,
    trials: int = 20
) -> dict

def to_float8(x: torch.Tensor, dtype=torch.float8_e4m3fn) -> tuple[torch.Tensor, torch.Tensor]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python benchmarks/kernels/benchmark_trtllm_decode_attention.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| dtype || torch.dtype || Base data type (typically bfloat16)
|-
| quant_dtypes || tuple || (q_dtype, kv_dtype, o_dtype) for mixed precision
|-
| batch_size || int || Number of sequences (1-256)
|-
| max_seq_len || int || Maximum sequence length (1K-131K)
|-
| num_heads || tuple[int, int] || (num_qo_heads, num_kv_heads), e.g., (64, 8) for GQA
|-
| head_size || int || Attention head dimension (default 128)
|-
| kv_layout || str || KV cache layout: "HND" or "NHD"
|-
| block_size || int || Paged attention block size (default 16)
|-
| warmup || int || Warmup iterations (default 10)
|-
| trials || int || Benchmark trials (default 20)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || torch.Tensor || Attention output, quantized if o_quant_dtype specified
|-
| benchmark_results || dict || Contains batch_size, mean/std times, speedup_percent, dtypes
|-
| results_csv || File || CSV file with all benchmark results
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Run full benchmark suite
python benchmarks/kernels/benchmark_trtllm_decode_attention.py

# Benchmark specific configuration
from benchmark_trtllm_decode_attention import benchmark_decode
import torch

FP8_DTYPE = torch.float8_e4m3fn
result = benchmark_decode(
    dtype=torch.bfloat16,
    quant_dtypes=(FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),  # All FP8
    batch_size=32,
    max_seq_len=8192,
    num_heads=(64, 8),  # GQA with 64 Q heads, 8 KV heads
    head_size=128,
    warmup=10,
    trials=20
)

print(f"TRT-LLM: {result['trtllm_mean']:.3f}ms")
print(f"Baseline: {result['baseline_mean']:.3f}ms")
print(f"Speedup: {result['speedup_percent']*100:.1f}%")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:FlashInfer_Attention]]
* [[Concept:Paged_Attention]]
* [[Concept:FP8_Quantization]]
* [[Benchmark:Attention_Performance]]
