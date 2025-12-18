{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Code Generation]], [[domain::Kernel Compilation]], [[domain::Quantization]], [[domain::GPTQ]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A code generator that produces optimized GPTQ-Marlin quantized GEMM kernel instantiations for multiple GPU architectures and quantization formats.

=== Description ===
This script generates CUDA kernel instantiations for GPTQ-Marlin quantization, supporting INT4 (AWQ/GPTQ), INT8, FP8, NVFP4, MXFP4, and HQQ quantization schemes with various activation types (FP16, BF16, INT8, FP8). It creates architecture-specific kernels for SM75 (Turing), SM80+ (Ampere/Ada), and SM89+ (Hopper with native FP8). The generator produces multiple kernel variants with different thread configurations, M-blocking factors, and group sizes for group-wise quantization. Unlike the MoE variant, this is for standard dense GEMM operations. It also generates a kernel selector header for runtime dispatch and supports float zero-points for HQQ quantization.

=== Usage ===
Run this script during the CMake build process to generate GPTQ-Marlin kernel variants for the target GPU architectures. The generated kernels are compiled into vLLM's main _C extension.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/csrc/quantization/gptq_marlin/generate_kernels.py csrc/quantization/gptq_marlin/generate_kernels.py]

=== Signature ===
<syntaxhighlight lang="python">
def remove_old_kernels() -> None
def generate_new_kernels() -> None

# Invoked with CUDA compute capabilities
# python generate_kernels.py "7.5,8.0,8.6,8.9,9.0"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run during build (invoked by CMakeLists.txt)
python csrc/quantization/gptq_marlin/generate_kernels.py "8.0,8.9"
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| sys.argv[1] || str || Comma-separated CUDA architectures (e.g., "7.5,8.0,8.9")
|-
| QUANT_CONFIGS || list[dict] || Quantization configurations with types and parameters
|-
| THREAD_CONFIGS || list[tuple] || Thread block configurations (K, N, threads)
|-
| THREAD_M_BLOCKS || list[float] || M-blocking factors (0.5, 1, 2, 3, 4)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| sm75_kernel_*.cu || File || Turing kernels (FP16/INT8 activations only)
|-
| sm80_kernel_*.cu || File || Ampere/Ada kernels (main variants)
|-
| sm89_kernel_*.cu || File || Hopper kernels with FP8 support
|-
| kernel_selector.h || File || Runtime dispatch based on dtype/config
|}

== Quantization Schemes ==

{| class="wikitable"
|-
! Scheme !! Weight Format !! Activation Types !! Group Blocks !! Special Features
|-
| AWQ-INT4 || kU4 || FP16, BF16 || -1, 2, 4, 8 || Standard AWQ
|-
| HQQ || kU4 || FP16 only || 4 || Float zero-points (is_zp_float=true)
|-
| GPTQ-INT4 || kU4B8 || FP16, BF16 || -1, 0, 2, 4, 8 || Bit-packed INT4
|-
| GPTQ-INT8 || kU8B128 || FP16, BF16 || -1, 0, 2, 4, 8 || Bit-packed INT8
|-
| FP8 || kFE4M3fn || FP16, BF16 || -1, 8 || Native FP8 weights
|-
| NVFP4 || kFE2M1f || FP16, BF16 || 1 || NVIDIA FP4 format
|-
| MXFP4 || kFE2M1f || BF16, FP8 || 2 || Microscaling FP4
|-
| INT8-Activation || Various || INT8 || -1, 2, 4, 8 || INT8 activations
|-
| FP8-Activation || Various || FP8 || -1, 2, 4, 8 || FP8 activations
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Generate kernels for all supported architectures
python csrc/quantization/gptq_marlin/generate_kernels.py \
    "7.5,8.0,8.6,8.9,9.0"

# Sample generated files:
# sm80_kernel_float16_u4_float16.cu - AWQ INT4 with FP16
# sm80_kernel_float16_u4_float16.cu (with is_zp_float) - HQQ variant
# sm80_kernel_bfloat16_u4b8_bfloat16.cu - GPTQ INT4 with BF16
# sm80_kernel_s8_u4_float16.cu - INT8 activation, INT4 weight
# sm89_kernel_fe4m3fn_u4b8_bfloat16.cu - FP8 activation, INT4 weight
# kernel_selector.h - Dispatch logic

# Kernel selector example:
# if (a_type == vllm::kFloat16 && b_type == vllm::kU4 &&
#     c_type == vllm::kFloat16 && group_blocks == 4 &&
#     is_zp_float == true)
#   kernel = Marlin<kFloat16, kU4, kFloat16, ..., true>;  # HQQ
# else if (a_type == vllm::kFE4M3fn)
#   TORCH_CHECK(false, "marlin kernel with fp8 activation is not built.");
</syntaxhighlight>

== Thread Configuration Heuristics ==

{| class="wikitable"
|-
! M Blocks !! Threads=256 Config !! Threads=128 Config
|-
| M <= 1 || (128, 128, 256) || (64, 128, 128) or (128, 64, 128)
|-
| M > 1 || (64, 256, 256) || Various
|}

== Related Pages ==
* [[Implementation:GPTQ_Quantization]]
* [[Implementation:Marlin_Kernels]]
* [[Concept:Group_Wise_Quantization]]
* [[Concept:HQQ_Quantization]]
* [[Build:Kernel_Code_Generation]]
