{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Kernel Benchmarking]], [[domain::Quantization]], [[domain::GEMM]], [[domain::Kernel Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
An auto-tuning benchmark for W8A8 block-wise FP8 matrix multiplication kernels optimized for DeepSeek-V3 and DeepSeek-R1 models.

=== Description ===
This benchmark auto-tunes Triton GEMM kernels for block-wise FP8 quantized matrix multiplication (W8A8) across various configurations. It searches through combinations of block sizes (BLOCK_M, BLOCK_N, BLOCK_K), group sizes, warp counts, and pipeline stages to find optimal kernel configurations for specific hardware and weight shapes. The tuner supports multi-GPU parallel tuning, distributing batch sizes across available GPUs for faster configuration search. Results are saved as JSON files containing best configurations per batch size and weight shape, specifically tuned for tensor-parallel DeepSeek architectures.

=== Usage ===
Use this tool to generate optimized kernel configurations for W8A8 FP8 inference on new hardware, tune performance for specific model architectures (especially DeepSeek variants), or when deploying with custom tensor parallelism configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_w8a8_block_fp8.py benchmarks/kernels/benchmark_w8a8_block_fp8.py]

=== Signature ===
<syntaxhighlight lang="python">
def w8a8_block_matmul(
    A: torch.Tensor, B: torch.Tensor,
    As: torch.Tensor, Bs: torch.Tensor,
    block_size: list[int],
    config: dict[str, Any],
    output_dtype: torch.dtype = torch.float16
) -> torch.Tensor

def tune(M: int, N: int, K: int, block_size: list[int],
         out_dtype: torch.dtype, search_space: list[dict],
         input_type: str) -> dict

def get_weight_shapes(tp_size: int) -> list[tuple[int, int]]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Tune for DeepSeek-V3 with TP=8
python benchmarks/kernels/benchmark_w8a8_block_fp8.py \
    --tp-size 8 --input-type fp8 --save-path ./configs/
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| A || torch.Tensor || FP8 activation matrix (M, K)
|-
| B || torch.Tensor || FP8 weight matrix (N, K)
|-
| As || torch.Tensor || Per-token-group scales for A, shape (M, K//block_k)
|-
| Bs || torch.Tensor || Per-block scales for B, shape (N//block_n, K//block_k)
|-
| block_size || list[int] || [block_n, block_k], typically [128, 128]
|-
| --tp-size || int || Tensor parallel size (default 8)
|-
| --block-n || int || Block size for N dimension (default 128)
|-
| --block-k || int || Block size for K dimension (default 128)
|-
| --batch-size || int || Single batch size to tune (optional)
|-
| --save-path || str || Directory to save tuned configs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| C || torch.Tensor || Output matrix (M, N) in output_dtype
|-
| config_json || File || Best kernel configs per batch size and weight shape
|-
| tuning_logs || stdout || Progress and timing information per configuration
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Auto-tune for DeepSeek-V3 with TP=8 across all GPUs
python benchmarks/kernels/benchmark_w8a8_block_fp8.py \
    --tp-size 8 \
    --input-type fp8 \
    --save-path ./model_executor/layers/quantization/utils/configs/

# Tune specific batch size on single GPU
python benchmarks/kernels/benchmark_w8a8_block_fp8.py \
    --tp-size 8 \
    --batch-size 64 \
    --block-n 128 \
    --block-k 128 \
    --save-path ./tuned_configs/

# Use tuned kernel
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm
)
import torch

M, N, K = 64, 4096, 7168
A = torch.randn(M, K, dtype=torch.float8_e4m3fn, device="cuda")
B = torch.randn(N, K, dtype=torch.float8_e4m3fn, device="cuda")
As = torch.randn(M, K//128, dtype=torch.float32, device="cuda")
Bs = torch.randn(N//128, K//128, dtype=torch.float32, device="cuda")

C = w8a8_triton_block_scaled_mm(A, B, As, Bs, [128, 128])
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Implementation:FP8_Quantization]]
* [[Concept:Block_Wise_Quantization]]
* [[Tool:Triton_Auto_Tuner]]
* [[Model:DeepSeek_V3]]
