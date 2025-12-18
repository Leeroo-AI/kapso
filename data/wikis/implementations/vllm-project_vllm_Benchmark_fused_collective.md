{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::Distributed]], [[domain::Kernel_Fusion]], [[domain::FlashInfer]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Benchmarks FlashInfer's fused collective operations (allreduce + rmsnorm + optional quantization) against standard separate operations for distributed LLM inference.

=== Description ===
This comprehensive benchmark compares FlashInfer's trtllm_allreduce_fusion operation against standard vLLM operations that perform allreduce, RMSNorm, and optional FP8/FP4 quantization as separate steps. The fused operation can combine up to four operations: tensor parallel all-reduce, residual addition, RMSNorm, and FP8/FP4 quantization, potentially reducing memory bandwidth requirements and improving performance.

The benchmark tests three fusion patterns: (1) allreduce + rmsnorm (no quantization), (2) allreduce + rmsnorm + FP8 quantization, and (3) allreduce + rmsnorm + FP4 quantization. For each pattern, it compares standard implementations (with both custom and native operators), torch.compiled versions, and FlashInfer's fused implementations in both oneshot and twoshot modes. Results are saved to markdown format with speedup calculations relative to the fastest baseline.

=== Usage ===
Run this benchmark when evaluating fused kernel benefits for distributed inference, comparing FlashInfer vs standard operations for tensor parallelism, or determining optimal quantization strategies for multi-GPU deployments.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_fused_collective.py benchmarks/kernels/benchmark_fused_collective.py]

=== Signature ===
<syntaxhighlight lang="python">
def run_benchmarks(
    num_tokens: int,
    hidden_dim: int,
    dtype: torch.dtype,
    use_residual: bool,
    allreduce_params: FlashInferFusedAllReduceParams | None,
    quant_modes: set[str],
    no_oneshot: bool,
) -> dict[str, float]
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run with 2 GPUs, default settings
torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_fused_collective.py

# Custom token counts and quantization modes
torchrun --nproc_per_node=4 benchmarks/kernels/benchmark_fused_collective.py \
    --num-tokens 128 512 1024 2048 \
    --quant-modes none,fp8,fp4 \
    --hidden-dim 8192

# Save results to file
torchrun --nproc_per_node=8 benchmarks/kernels/benchmark_fused_collective.py \
    --output-file benchmark_results.md \
    --dtypes bfloat16 \
    --no-oneshot

# Test without residual connections
torchrun --nproc_per_node=2 benchmarks/kernels/benchmark_fused_collective.py \
    --no-residual \
    --quant-modes fp8
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| num_tokens || list[int] || Token counts to benchmark (default: 128, 512, 1024, 2048)
|-
| hidden_dim || int || Hidden dimension size (default: 8192)
|-
| dtypes || list[str] || Data types to test (float16, bfloat16, float32)
|-
| quant_modes || str || Quantization modes: none, fp8, fp4 (comma-separated)
|-
| no_residual || bool || Skip residual connection tests
|-
| warmup || int || Warmup iterations (default: 5)
|-
| trials || int || Benchmark trials (default: 20)
|-
| output_file || str || Markdown output file path
|-
| no_oneshot || bool || Skip oneshot mode benchmarks
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| results || dict[str, float] || Time in ms for each operation variant
|-
| speedup || float || Speedup vs fastest baseline
|-
| markdown_report || str || Formatted markdown tables with all results
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# FlashInfer fused allreduce + rmsnorm
def flashinfer_fused_allreduce_rmsnorm(
    input_tensor, residual, rms_gamma, rms_eps,
    allreduce_params, use_oneshot, norm_out=None
):
    flashinfer_comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        token_num=input_tensor.shape[0],
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        hidden_dim=input_tensor.shape[-1],
        workspace_ptrs=_FI_WORKSPACE_TENSOR,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        use_oneshot=use_oneshot,
        **allreduce_params.get_trtllm_fused_allreduce_kwargs()
    )

# Standard vLLM operations
class VllmFusedAllreduce:
    def allreduce_rmsnorm(self, input_tensor, residual):
        allreduce_out = tensor_model_parallel_all_reduce(input_tensor)
        return self.rms_norm(allreduce_out, residual)

    def allreduce_rmsnorm_fp8_quant(self, input_tensor, residual, scale_factor):
        allreduce_out = tensor_model_parallel_all_reduce(input_tensor)
        rms_out = self.rms_norm(allreduce_out, residual)
        return self.fp8_quant(rms_out, scale_factor)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:benchmark_device_communicators]]
* [[Concept:Kernel_Fusion]]
* [[Concept:Tensor_Parallelism]]
* [[Concept:FlashInfer]]
