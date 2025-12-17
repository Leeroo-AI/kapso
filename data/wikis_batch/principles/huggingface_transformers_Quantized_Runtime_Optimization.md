{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
* [[source::Paper|CUDA Kernels|https://developer.nvidia.com/cuda-zone]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]], [[domain::GPU Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Configuring and utilizing optimized computational kernels for efficient execution of quantized model operations.

=== Description ===

Runtime optimization for quantized models involves selecting and configuring specialized kernels that accelerate quantized operations. This includes:

* '''Kernel Selection:''' Choosing from available backends (CUDA, Triton, vendor-optimized)
* '''Fused Operations:''' Combining dequantization with matrix multiplication
* '''Memory Access Patterns:''' Optimizing for quantized data layouts
* '''Hardware Utilization:''' Leveraging tensor cores and specialized instructions

Different quantization methods and hardware combinations require different kernel strategies. Modern GPUs (H100, A100) have hardware-accelerated support for low-precision operations.

=== Usage ===

Runtime optimizations are configured automatically based on quantization method, hardware, and available backends. Users can influence selection through backend parameters in the quantization config. Performance gains range from 2-4x depending on method and hardware.

== Theoretical Basis ==

=== Fused Dequant-MatMul ===

Standard approach (inefficient):
<math>
\begin{align}
W_{fp} &= \text{dequant}(W_{quant}, S) \\
Y &= W_{fp} \cdot X
\end{align}
</math>

Fused approach (optimized):
<math>
Y = \text{dequant\_matmul}(W_{quant}, S, X)
</math>

The fused kernel:
1. Loads quantized weights from memory
2. Dequantizes on-the-fly in registers
3. Accumulates products immediately
4. Never materializes full <math>W_{fp}</math>

Memory savings: <math>\text{MemAccess} = |W_{quant}| + |S|</math> vs. <math>|W_{fp}|</math>

=== Tensor Core Utilization ===

Modern GPUs have specialized units for low-precision operations:

'''INT8 Tensor Cores (A100):'''
* Operate on 8-bit integers
* Accumulate in INT32
* Throughput: 312 TOPS (INT8) vs. 156 TFLOPS (FP16)

'''FP8 Tensor Cores (H100):'''
* E4M3 or E5M2 format
* Hardware-accelerated conversion
* Throughput: 989 TFLOPS (FP8) vs. 494 TFLOPS (BF16)

Optimal utilization requires:
* Properly shaped tensors (multiples of 8/16)
* Aligned memory access
* Efficient data layout

=== Block-wise Processing ===

For group-wise quantization:

<math>
Y = \sum_{g=1}^{G} S_g \cdot (W_{g,quant} \cdot X_g)
</math>

Efficient kernel processes blocks:
<syntaxhighlight lang="text">
for each block in parallel:
    load W_quant[block] → shared memory
    load S[block] → registers
    for each element in block:
        dequant = scale * W_quant[i]
        accumulate += dequant * X[i]
</syntaxhighlight>

Block size affects:
* Memory coalescing
* Register pressure
* Occupancy

=== Memory Layout Optimization ===

'''Standard Layout (row-major):'''
<math>
W[i,j] = W[i \cdot n + j]
</math>

'''Quantized Pack Layout:'''
<math>
W_{packed}[i,j/2] = (W[i,j] \ll 4) | W[i,j+1]
</math>

Two 4-bit values per byte improves:
* Memory bandwidth (2x fewer loads)
* Cache efficiency (2x effective cache size)
* Occupancy (more threads fit)

=== Backend-Specific Optimizations ===

'''Marlin (GPTQ/AWQ):'''
* Optimized for group-wise 4-bit
* Fused dequant-matmul
* Supports A100/H100 tensor cores
* 3-4x speedup over baseline

'''Exllama (GPTQ):'''
* Custom CUDA kernels
* Specialized for act-order
* Supports variable group sizes
* Good for consumer GPUs

'''cuBLAS INT8 (BnB):'''
* Uses NVIDIA cuBLAS
* INT8 tensor core matmul
* Fallback for outliers
* Robust across GPUs

'''Triton (Various):'''
* Python-based kernel language
* JIT compilation
* Flexible but may be slower than hand-tuned CUDA

== Performance Characteristics ==

=== Speedup Analysis ===

Theoretical speedup from quantization:

<math>
S_{theoretical} = \frac{b_{orig}}{b_{quant}} \times \frac{T_{orig}}{T_{quant}}
</math>

Where:
* <math>b</math> is bit-width
* <math>T</math> is compute time per operation

Practical speedup is lower due to:
* Dequantization overhead
* Memory bandwidth limits
* Kernel launch overhead

=== Memory Bandwidth Bottleneck ===

For memory-bound operations:

<math>
T_{compute} = \frac{\text{FLOPs}}{\text{TFLOPS}} < T_{memory} = \frac{\text{Bytes}}{\text{BW}}
</math>

Quantization helps when memory-bound by reducing Bytes:

<math>
\text{BW}_{effective} = \text{BW}_{physical} \times \frac{b_{orig}}{b_{quant}}
</math>

=== Batch Size Effects ===

Quantization benefits vary with batch size:

* '''Small batch''' (1-4): Reduced memory bandwidth dominant, 3-4x speedup
* '''Medium batch''' (8-32): Mix of memory and compute benefits, 2-3x speedup
* '''Large batch''' (64+): Compute-bound, 1.5-2x speedup

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_quantizer_runtime_config]]

=== Requires ===
* [[requires::Principle:huggingface_transformers_Quantized_Weight_Loading]]

=== Depends On ===
* [[depends_on::Environment:huggingface_transformers_CUDA]]
