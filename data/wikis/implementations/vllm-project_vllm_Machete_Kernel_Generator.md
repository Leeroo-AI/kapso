{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Code Generation]], [[domain::CUTLASS]], [[domain::Quantization]], [[domain::GEMM]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A comprehensive code generator for Machete quantized GEMM kernels using CUTLASS 3.x templates with heuristic-based kernel selection.

=== Description ===
This sophisticated code generator produces CUTLASS 3.x-based Machete kernels for mixed-precision quantized GEMM operations, supporting GPTQ (U4B8, U8B128) and AWQ (U4, U8) quantization with group-wise scales and zero-points. It generates three types of files: dispatch logic with type-based routing, implementation files with kernel instantiations (split across 8 files for faster compilation), and prepack utilities for weight layout conversion. The generator uses M/N/K-dependent heuristics tuned for H100 GPUs to select optimal tile shapes (128x128, 128x256, 256x128, etc.) and cluster configurations. It supports StreamK scheduling for load balancing and TMA (Tensor Memory Accelerator) for efficient memory access.

=== Usage ===
Run this script during CMake configuration to generate Machete kernel sources for CUTLASS-based quantized GEMM. The generated kernels provide high-performance alternatives to Marlin for certain quantization schemes and hardware configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/csrc/quantization/machete/generate.py csrc/quantization/machete/generate.py]

=== Signature ===
<syntaxhighlight lang="python">
@dataclass(frozen=True)
class ScheduleConfig:
    tile_shape_mn: tuple[int, int]
    cluster_shape_mnk: tuple[int, int, int]
    kernel_schedule: MixedInputKernelScheduleType
    epilogue_schedule: EpilogueScheduleType
    tile_scheduler: TileSchedulerType

@dataclass(frozen=True)
class TypeConfig:
    a: DataType  # Activation
    b: DataType | VLLMDataType  # Weight
    b_group_scale: DataType
    b_group_zeropoint: DataType
    b_channel_scale: DataType
    a_token_scale: DataType
    out: DataType
    accumulator: DataType

def generate() -> None
def create_sources(impl_configs: list[ImplConfig], num_impl_files=8) -> list[tuple[str, str]]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run during build
python csrc/quantization/machete/generate.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| TypeConfig || dataclass || Defines dtypes for all GEMM components
|-
| ScheduleConfig || dataclass || Defines tile/cluster shapes and schedules
|-
| heuristic || dict || M/N/K-based conditions for kernel selection
|-
| num_impl_files || int || Number of implementation files to generate (default 8)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| machete_mm_dispatch.cu || File || Type-based dispatcher to appropriate kernels
|-
| machete_mm_impl_part*.cu || File || Split implementation files (8 parts for parallel compilation)
|-
| machete_prepack.cu || File || Weight prepacking utilities for optimal layout
|}

== Kernel Selection Heuristics ==

{| class="wikitable"
|-
! M Range !! K/N Conditions !! Tile Shape !! Cluster Shape
|-
| M > 256 || K <= 16384, N <= 4096 || (128, 128) || (2, 1, 1)
|-
| M > 256 || Otherwise || (128, 256) || (2, 1, 1)
|-
| M > 128 || K <= 4096, N <= 4096 || (128, 64) || (2, 1, 1)
|-
| M > 128 || K <= 8192, N <= 8192 || (128, 128) || (2, 1, 1)
|-
| M > 64 || K >= 8192, N >= 12288 || (256, 128) || (2, 1, 1)
|-
| M > 32 || K >= 16384, N >= 12288 || (256, 64) || (2, 1, 1)
|-
| M > 16 || K <= 12288, N <= 8192 || (128, 32) || (2, 1, 1)
|-
| M <= 16 || N >= 26624 || (256, 16) || (1, 1, 1)
|-
| Default || - || (128, 16) || (1, 1, 1)
|}

== Supported Quantization Types ==

{| class="wikitable"
|-
! Format !! Weight Type !! Activation !! Scales !! Zero-Points
|-
| GPTQ-INT4 || u4b8 || FP16/BF16 || Group-wise (FP16/BF16) || None
|-
| GPTQ-INT8 || u8b128 || FP16/BF16 || Group-wise (FP16/BF16) || None
|-
| AWQ-INT4 || u4 || FP16/BF16 || Group-wise (FP16/BF16) || Group-wise (FP16/BF16)
|-
| AWQ-INT8 || u8 || FP16/BF16 || Group-wise (FP16/BF16) || Group-wise (FP16/BF16)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Generate Machete kernels during build
python csrc/quantization/machete/generate.py

# Output structure:
# csrc/quantization/machete/generated/
#   ├── machete_mm_dispatch.cu
#   ├── machete_mm_impl_part1.cu
#   ├── machete_mm_impl_part2.cu
#   ├── ...
#   ├── machete_mm_impl_part8.cu
#   └── machete_prepack.cu

# Dispatch logic example:
# if (b_type == vllm::ScalarType::U4B8 &&
#     a_type == torch::kFloat16 &&
#     out_type == torch::kFloat16 &&
#     maybe_g_scales_type == torch::kFloat16 &&
#     !maybe_g_zeros_type)
#   return mm_dispatch_u4b8_f16_f16(args);

# Kernel instantiation example:
# template __global__ void MacheteKernelTemplate<
#   cutlass::half_t,        // ElementA
#   vllm::U4B8,            // ElementB
#   cutlass::half_t,        // ElementD
#   float,                  // Accumulator
#   cutlass::half_t,        // GroupScaleT
#   void,                   // GroupZeroT
#   void,                   // ChannelScaleT
#   void,                   // TokenScaleT
#   cutlass::gemm::KernelTmaWarpSpecializedCooperative,
#   sch_128x128_2x1x1_TmaMI_TmaCoop_StreamKScheduler
# >(MACHETE_KERNEL_PARAMS);
</syntaxhighlight>

== Related Pages ==
* [[Implementation:CUTLASS_GEMM]]
* [[Implementation:Machete_Quantization]]
* [[Concept:Tile_Scheduling]]
* [[Concept:TMA_Memory_Access]]
* [[Tool:CUTLASS_3]]
