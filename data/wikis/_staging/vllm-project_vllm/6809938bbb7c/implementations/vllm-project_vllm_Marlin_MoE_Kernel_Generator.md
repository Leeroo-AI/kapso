{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Code Generation]], [[domain::Kernel Compilation]], [[domain::Quantization]], [[domain::MoE]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A code generator that produces optimized Marlin MoE quantized GEMM kernel instantiations for multiple GPU architectures and quantization schemes.

=== Description ===
This script generates CUDA kernel instantiations for Marlin MoE (Mixture-of-Experts) with W16A4 quantization, supporting AWQ, GPTQ, FP8, NVFP4, and MXFP4 quantization formats. It produces architecture-specific kernels for SM75 (Turing), SM80+ (Ampere/Ada), and SM89+ (Hopper with FP8 support). The generator uses Jinja2 templates to create kernel files with various thread block configurations (128x128x256, 64x256x256, 64x128x128), thread-M-blocks (0.5-4), and group sizes for group-wise quantization. It also generates a kernel selector header for runtime dispatch based on configuration parameters.

=== Usage ===
Run this script during the build process to generate Marlin MoE kernel variants for the target GPU architectures. The generated kernels are compiled into vLLM's _moe_C extension.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/csrc/moe/marlin_moe_wna16/generate_kernels.py csrc/moe/marlin_moe_wna16/generate_kernels.py]

=== Signature ===
<syntaxhighlight lang="python">
def remove_old_kernels() -> None
def generate_new_kernels() -> None

# Called with CUDA architectures as argument
# python generate_kernels.py "8.0,8.6,8.9,9.0"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run during CMake build (called by CMakeLists.txt)
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.0,8.6,8.9,9.0"
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| sys.argv[1] || str || Comma-separated CUDA compute capabilities (e.g., "8.0,8.9,9.0")
|-
| QUANT_CONFIGS || list[dict] || Quantization scheme configurations (AWQ, GPTQ, FP8, etc.)
|-
| THREAD_CONFIGS || list[tuple] || Thread block shapes (K, N, threads)
|-
| THREAD_M_BLOCKS || list[float] || M-dimension blocking factors
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| sm75_kernel_*.cu || File || Turing (SM75) kernels for FP16 activations
|-
| sm80_kernel_*.cu || File || Ampere+ (SM80+) kernels, main variants
|-
| sm89_kernel_*.cu || File || Hopper (SM89+) kernels with FP8 support
|-
| kernel_selector.h || File || Runtime dispatch logic based on config parameters
|}

== Kernel Configurations ==

{| class="wikitable"
|-
! Quantization !! Weight Type !! Activation Types !! Group Sizes
|-
| AWQ-INT4 || kU4 || FP16, BF16, INT8, FP8 || -1, 2, 4, 8
|-
| GPTQ-INT4 || kU4B8 || FP16, BF16, INT8, FP8 || -1, 0, 2, 4, 8
|-
| AWQ-INT8 || kU8B128 || FP16, BF16 || -1, 0, 2, 4, 8
|-
| FP8 || kFE4M3fn || FP16, BF16 || -1, 8
|-
| NVFP4 || kFE2M1f || FP16, BF16 || 1
|-
| MXFP4 || kFE2M1f || BF16, FP8 || 2
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Generate kernels for Ampere and Hopper
python csrc/moe/marlin_moe_wna16/generate_kernels.py "8.0,8.6,8.9"

# Output files include:
# - sm80_kernel_float16_u4_float16.cu (AWQ-INT4 with FP16)
# - sm80_kernel_bfloat16_u4b8_bfloat16.cu (GPTQ-INT4 with BF16)
# - sm89_kernel_fe4m3fn_u4b8_bfloat16.cu (FP8 activation, INT4 weight)
# - kernel_selector.h (dispatch logic)

# The kernel selector generates code like:
# if (a_type == vllm::kFloat16 && b_type == vllm::kU4 &&
#     c_type == vllm::kFloat16 && threads == 256 &&
#     thread_m_blocks == 2 && group_blocks == 4)
#   kernel = Marlin<...template params...>;
# else if (...)
#   kernel = Marlin<...>;
</syntaxhighlight>

== Related Pages ==
* [[Implementation:Marlin_MoE_Kernels]]
* [[Concept:W4A16_Quantization]]
* [[Tool:Jinja2_Code_Generation]]
* [[Build:CMake_CUDA_Kernels]]
* [[Concept:MoE_Quantization]]
