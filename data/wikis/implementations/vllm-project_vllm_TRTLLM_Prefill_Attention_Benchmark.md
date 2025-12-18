{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Kernel Benchmarking]], [[domain::Attention Mechanisms]], [[domain::Quantization]], [[domain::Prefill Phase]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A benchmark comparing TensorRT-LLM prefill attention kernels against FlashInfer baseline with mixed-precision quantization support.

=== Description ===
This benchmark evaluates TensorRT-LLM's prefill attention implementation (for prompt processing phase) with support for FP8/FP4 quantization of queries, KV cache, and outputs. Unlike decode attention which processes single tokens, prefill attention handles entire input prompts efficiently. The benchmark tests performance across batch sizes (1-256) and sequence lengths (1K-131K), supporting GQA (Grouped Query Attention) with configurable query/KV head ratios. Results include speedup percentages relative to FlashInfer baseline and are saved to CSV.

=== Usage ===
Use this benchmark to evaluate TRT-LLM prefill attention performance for long-context inference, compare quantization strategies for prompt processing, or optimize first-token latency with mixed-precision attention.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_trtllm_prefill_attention.py benchmarks/kernels/benchmark_trtllm_prefill_attention.py]

=== Signature ===
<syntaxhighlight lang="python">
def benchmark_prefill(
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
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python benchmarks/kernels/benchmark_trtllm_prefill_attention.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| dtype || torch.dtype || Base data type (typically bfloat16)
|-
| quant_dtypes || tuple || (q_dtype, kv_dtype, o_dtype) - None means use base dtype
|-
| batch_size || int || Number of sequences to process
|-
| max_seq_len || int || Maximum sequence length (prompt length)
|-
| num_heads || tuple || (num_qo_heads, num_kv_heads) for GQA support
|-
| head_size || int || Dimension of each attention head
|-
| kv_layout || str || KV cache memory layout
|-
| block_size || int || Paged attention block size
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| attention_output || torch.Tensor || Attention result for entire prompt
|-
| timing_results || dict || Mean/std times for TRT-LLM and baseline
|-
| speedup_percent || float || Performance improvement over baseline
|-
| csv_results || File || Comprehensive benchmark data
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Run full benchmark
python benchmarks/kernels/benchmark_trtllm_prefill_attention.py

# Custom configuration
from benchmark_trtllm_prefill_attention import benchmark_prefill
import torch

result = benchmark_prefill(
    dtype=torch.bfloat16,
    quant_dtypes=(torch.float8_e4m3fn, torch.float8_e4m3fn, None),
    batch_size=8,
    max_seq_len=16384,  # Long context
    num_heads=(64, 8),
    head_size=128
)

print(f"Prefill speedup: {result['speedup_percent']*100:.1f}%")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_trtllm_decode_attention.py]]
* [[Concept:Prefill_vs_Decode]]
* [[Benchmark:Long_Context_Performance]]
