# Benchmark MoE Kernels

**Knowledge Sources:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_moe.py`
**Domains:** Performance Testing, Kernel Benchmarking, Mixture of Experts, Triton Tuning
**Last Updated:** 2025-12-17

## Overview

A comprehensive benchmarking and auto-tuning framework for Mixture of Experts (MoE) fused kernels supporting multiple quantization schemes and tensor parallel configurations.

## Description

The `benchmark_moe.py` script provides both benchmarking and auto-tuning capabilities for fused MoE operations in vLLM. It supports a wide range of MoE architectures including Mixtral, Qwen2MoE, DeepSeek-V2/V3, DBRX, Jamba, and others. The script can measure performance of existing kernel configurations or automatically tune Triton kernel parameters to find optimal settings for specific hardware and model configurations.

**Core Functionality:**

1. **Auto-Tuning Mode (`--tune`)**: Searches through a large configuration space of Triton kernel parameters (block sizes, warps, stages, group sizes) to find the optimal configuration for each batch size. Results are saved as JSON configuration files for production use.

2. **Benchmarking Mode (default)**: Loads existing optimal configurations and measures MoE kernel performance across different batch sizes, reporting latency in microseconds.

**MoE Operations Covered:**
- `fused_topk`: Expert selection and weight computation
- `fused_experts`: Combined permute, expert computation (w1, silu_and_mul, w2), and unpermute operations
- Support for FP8 W8A8, INT8 W8A16, and FP16 configurations
- Optional DeepGEMM integration for block-quantized FP8

**Quantization Support:**
- **FP8 W8A8**: Weight and activation quantization with per-tensor or block-wise scaling
- **INT8 W8A16**: 8-bit weight, 16-bit activation quantization
- **FP16**: Full precision baseline
- Block quantization: Configurable block shapes for fine-grained scaling

**Distributed Support:**
- Tensor Parallelism (TP): Shards intermediate_size across TP ranks
- Expert Parallelism (EP): Distributes experts across TP ranks
- Proper dimension calculations for both modes

**Platform Optimizations:**
- CUDA: Focused search space with practical parameter ranges
- ROCm: Extended search space with architecture-specific pruning
- Device-specific optimizations and memory layout considerations

## Usage

The script supports both tuning and benchmarking modes with extensive configuration options.

**Basic Usage:**

```bash
# Benchmark existing configurations for Mixtral-8x7B
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2

# Tune optimal configurations and save to directory
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --tune \
    --save-dir ./moe_configs

# Benchmark with FP8 quantization
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --dtype fp8_w8a8

# Benchmark with INT8 quantization
python benchmarks/kernels/benchmark_moe.py \
    --model deepseek-ai/DeepSeek-V2 \
    --tp-size 4 \
    --dtype int8_w8a16

# Benchmark specific batch sizes
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --batch-size 256 512 1024

# Benchmark with Expert Parallelism
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --enable-expert-parallel

# Benchmark with DeepGEMM (FP8 block quantization)
python benchmarks/kernels/benchmark_moe.py \
    --model deepseek-ai/DeepSeek-V3 \
    --tp-size 8 \
    --dtype fp8_w8a8 \
    --use-deep-gemm
```

**Command-line Arguments:**

- `--model`: HuggingFace model name (default: mistralai/Mixtral-8x7B-Instruct-v0.1)
- `--tp-size` / `-tp` / `--tensor-parallel-size`: Tensor parallel size (default: 2)
- `--enable-expert-parallel` / `-enable-ep`: Enable expert parallelism mode
- `--dtype`: Quantization type - auto, fp8_w8a8, int8_w8a16 (default: auto)
- `--use-deep-gemm`: Enable DeepGEMM for FP8 block quantization
- `--save-dir`: Directory to save tuned configurations (default: ./)
- `--seed`: Random seed for reproducibility (default: 0)
- `--batch-size`: Specific batch sizes to test (default: comprehensive range)
- `--tune`: Enable auto-tuning mode
- `--trust-remote-code`: Allow loading models with custom code
- `--model-prefix`: Config attribute prefix for nested configs

## Code Reference

**Source Location:** `/tmp/praxium_repo_583nq7ea/benchmarks/kernels/benchmark_moe.py`

**Core Benchmark Function:**

```python
def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 100,
    block_quant_shape: list[int] = None,
    use_deep_gemm: bool = False,
) -> float
```

**Ray Worker for Distributed Benchmarking:**

```python
@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        block_quant_shape: list[int] = None,
        use_deep_gemm: bool = False,
    ) -> tuple[dict[str, int], float]

    def tune(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        search_space: list[dict[str, int]],
        block_quant_shape: list[int],
        use_deep_gemm: bool,
    ) -> dict[str, int]
```

**Configuration Generation:**

```python
def get_configs_compute_bound(use_fp16, block_quant_shape) -> list[dict[str, int]]
```

**Entry Point:**

```python
def main(args: argparse.Namespace) -> None
```

**Import:**

```python
# Run as script
python benchmarks/kernels/benchmark_moe.py [OPTIONS]
```

## I/O Contract

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `model` | str | HuggingFace model identifier for MoE architecture |
| `tp_size` | int | Tensor parallel size for distributed computation |
| `enable_expert_parallel` | bool | Whether to use expert parallelism instead of TP |
| `dtype` | str | Quantization scheme: auto, fp8_w8a8, int8_w8a16 |
| `use_deep_gemm` | bool | Enable DeepGEMM for block-quantized FP8 |
| `batch_size` | list[int] | Specific batch sizes to benchmark |
| `tune` | bool | Enable auto-tuning mode |
| `save_dir` | str | Directory for saving tuned configurations |
| `seed` | int | Random seed for reproducibility |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| Console Output | str | Benchmark results showing batch size, config, and kernel time |
| Config Files | JSON | Tuned kernel configurations saved to save_dir (tuning mode) |
| Latency | float | Kernel execution time in microseconds |

### Configuration Parameters (Triton Tuning)

| Parameter | Type | Description |
|-----------|------|-------------|
| `BLOCK_SIZE_M` | int | Block size for M dimension (batch) |
| `BLOCK_SIZE_N` | int | Block size for N dimension (output features) |
| `BLOCK_SIZE_K` | int | Block size for K dimension (input features) |
| `GROUP_SIZE_M` | int | Number of M blocks to group together |
| `num_warps` | int | Number of warps per thread block |
| `num_stages` | int | Number of pipeline stages for instruction overlap |
| `waves_per_eu` | int | ROCm-specific: waves per execution unit |
| `matrix_instr_nonkdim` | int | ROCm-specific: matrix instruction size |
| `kpack` | int | ROCm-specific: K-dimension packing factor |

## Usage Examples

### Example 1: Quick Benchmark

```python
# Benchmark Mixtral-8x7B with default settings
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --batch-size 256 512 1024

# Output:
# Batch size: 256, config: {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, ...}
# Kernel time: 145.32 us
# Batch size: 512, config: {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, ...}
# Kernel time: 267.89 us
# Batch size: 1024, config: {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, ...}
# Kernel time: 512.45 us
```

### Example 2: Auto-Tuning for Production

```python
# Tune optimal configurations for DeepSeek-V2 with FP8
python benchmarks/kernels/benchmark_moe.py \
    --model deepseek-ai/DeepSeek-V2 \
    --tp-size 4 \
    --dtype fp8_w8a8 \
    --tune \
    --save-dir ./configs/deepseek_v2_fp8 \
    --seed 42

# This will:
# 1. Search through ~1000+ Triton configurations per batch size
# 2. Test batch sizes: [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
# 3. Save optimal configs to JSON files like:
#    ./configs/deepseek_v2_fp8/E160_N7168_dtype_float8_e4m3fn.json
# 4. Each file maps batch_size -> optimal_config
```

### Example 3: FP8 Block Quantization with DeepGEMM

```python
# Benchmark DeepSeek-V3 with block-quantized FP8
python benchmarks/kernels/benchmark_moe.py \
    --model deepseek-ai/DeepSeek-V3 \
    --tp-size 8 \
    --dtype fp8_w8a8 \
    --use-deep-gemm \
    --batch-size 1024 2048 4096

# DeepGEMM features:
# - Block shape: [128, 128] (default for DeepGEMM)
# - Per-block scaling factors for weights
# - Improved accuracy over per-tensor scaling
# - Optimized for large batch inference
```

### Example 4: Expert Parallelism

```python
# Benchmark with expert parallelism for Mixtral
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 4 \
    --enable-expert-parallel \
    --batch-size 512 1024

# Expert Parallelism characteristics:
# - Distributes 8 experts across 4 GPUs (2 experts/GPU)
# - Full intermediate_size per GPU (no sharding)
# - Better load balancing for expert utilization
# - Different communication patterns vs TP
```

### Example 5: INT8 Weight Quantization

```python
# Benchmark INT8 W8A16 quantization
python benchmarks/kernels/benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --dtype int8_w8a16 \
    --batch-size 256 512 1024 2048

# INT8 W8A16 characteristics:
# - 8-bit integer weights with per-channel or per-group scaling
# - 16-bit activations (no activation quantization)
# - Good balance between performance and accuracy
# - Lower memory bandwidth than FP16
# - Typical speedup: 1.5-2x over FP16
```

### Example 6: Comprehensive Tuning Sweep

```python
# Full tuning across all default batch sizes
python benchmarks/kernels/benchmark_moe.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 4 \
    --tune \
    --save-dir ./configs/qwen2_moe \
    --trust-remote-code

# This performs:
# - Tuning for 18 batch sizes
# - ~20 iterations per config (faster but less accurate)
# - ROCm-specific pruning if on AMD GPUs
# - Saves per-model, per-dtype configuration files
# - Total time: ~30-60 minutes depending on hardware
```

### Example 7: Analyzing Tuned Configurations

```python
# After tuning, configuration files contain:
{
  "triton_version": "2.1.0",
  "1": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3
  },
  "256": {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 16,
    "num_warps": 8,
    "num_stages": 4
  },
  ...
}

# Observations:
# - Small batches prefer smaller block sizes and fewer warps
# - Large batches benefit from larger blocks and more parallelism
# - More pipeline stages (num_stages) help hide memory latency
# - GROUP_SIZE_M affects work distribution across SMs
```

## Implementation Notes

**Supported Model Architectures:**
- **Mixtral**: num_local_experts, num_experts_per_tok, intermediate_size
- **Qwen2MoE/Qwen3MoE**: num_experts, num_experts_per_tok, moe_intermediate_size
- **DeepSeek-V2/V3**: n_routed_experts, num_experts_per_tok, moe_intermediate_size
- **DBRX**: ffn_config.moe_num_experts, ffn_config.moe_top_k
- **Jamba**: num_experts, num_experts_per_tok
- **Llama4**: get_text_config(), num_local_experts
- **GLM4MoE**: n_routed_experts, num_experts_per_tok
- **NemotronH**: n_routed_experts, num_experts_per_tok
- **HunYuanMoE**: num_experts, moe_topk[0], moe_intermediate_size[0]

**Search Space Sizes:**
- **CUDA**: ~500-800 configurations (block_m: 5, block_n: 4, block_k: 3, warps: 2, group_m: 4, stages: 4)
- **ROCm**: ~2000-5000 configurations before pruning (adds waves_per_eu, matrix_instr_nonkdim, kpack)
- **ROCm with pruning**: ~500-1000 configurations after architecture-specific filtering

**CUDA Graph Optimization:**
- Captures 10 consecutive kernel invocations in a single graph
- Reduces launch overhead significantly
- Latency measured as average over 10 invocations per graph replay
- Important for accurate performance measurement at small batch sizes

**Block Quantization Constraints:**
- BLOCK_SIZE_K must be multiple of block_k
- BLOCK_SIZE_N must be multiple of block_n
- Default block shape for DeepGEMM: [128, 128]
- Configurations violating constraints are filtered out

**ROCm-Specific Optimizations:**
- Architecture detection (MI200/MI300: sm_version 90+, MI100: sm_version 80+)
- MFMA instruction size selection (16 or 32)
- LDS memory constraint checking (max 64KB)
- Wave-per-EU tuning for occupancy
- Different pruning heuristics for small vs large GEMMs

**Memory Considerations:**
- MoE kernels are typically memory-bandwidth bound
- FP8 reduces bandwidth by 2x vs FP16
- INT8 reduces bandwidth by 2x vs FP16
- Block quantization adds overhead but improves accuracy
- Expert parallelism reduces memory per GPU but increases communication

**Performance Characteristics:**
- Small batches (< 32): Latency-sensitive, prefer low-overhead configs
- Medium batches (32-512): Balanced, trade-off between occupancy and memory
- Large batches (> 512): Throughput-focused, maximize parallelism and pipelining
- FP8 W8A8: 1.5-2.5x speedup over FP16, higher at large batches
- INT8 W8A16: 1.3-2x speedup over FP16
- DeepGEMM: Similar speed to standard FP8 but better accuracy

**Tuning Best Practices:**
- Use consistent hardware for tuning and inference
- Tune with representative batch size distribution
- Consider warmup effects in production (tuning only runs 20 iters)
- Save multiple config sets for different hardware generations
- Validate accuracy after tuning (not checked by benchmark)

**Distributed Execution:**
- Uses Ray for multi-GPU tuning/benchmarking
- Automatically detects available GPUs
- Distributes workload round-robin across GPUs
- ROCm requires ROCR_VISIBLE_DEVICES (auto-handled)

## Related Pages

- **vllm Fused MoE Layers** - MoE layer implementations using these kernels
- **vllm MoE Quantization** - Quantization strategies for MoE models
- **vllm Triton Kernels** - Triton kernel implementations
- **vllm Expert Parallelism** - Expert parallelism implementation details
- **vllm Tensor Parallelism** - Tensor parallelism for MoE models
- **vllm CUDA Graph** - CUDA Graph integration for kernel optimization
- **vllm FP8 Support** - FP8 quantization infrastructure
- **vllm DeepGEMM** - Block-quantized FP8 GEMM implementation

**See Also:**
- https://github.com/vllm-project/vllm - vLLM repository
- Mixtral architecture and MoE design
- Triton compiler documentation
- ROCm/CUDA performance tuning guides
