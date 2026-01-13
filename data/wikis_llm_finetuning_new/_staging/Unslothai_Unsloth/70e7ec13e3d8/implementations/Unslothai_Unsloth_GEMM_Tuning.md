# GEMM Manual Tuning Configurations

[[domain::Kernels]] [[domain::MoE]] [[domain::GPU_Optimization]]

## Overview

The GEMM Tuning module (`unsloth/kernels/moe/grouped_gemm/kernels/tuning.py`) provides manual tuning utilities and configuration dataclasses for grouped GEMM kernels. It enables fine-grained control over kernel parameters when autotuning is not desired or for reproducing specific configurations.

## Purpose

This module supports manual kernel optimization by:

1. **Defining configuration dataclasses** for forward and backward kernels
2. **Providing device property queries** for hardware-aware tuning
3. **Implementing configuration pruning** based on hardware constraints
4. **Offering result tracking and comparison utilities** for benchmarking

## Key Components

### Device Properties

#### `DeviceProperties` Dataclass

Captures GPU hardware characteristics:

```python
@dataclass
class DeviceProperties:
    NUM_SM: int        # Number of streaming multiprocessors
    NUM_REGS: int      # Maximum registers per SM
    SIZE_SMEM: int     # Maximum shared memory per SM
    WARP_SIZE: int     # Threads per warp (typically 32)
```

#### `get_device_properties`

Queries and caches device properties:

```python
_DEVICE_PROPERTIES: Optional[DeviceProperties] = None

def get_device_properties():
    global _DEVICE_PROPERTIES
    if _DEVICE_PROPERTIES is None:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device()
        )
        NUM_SM = properties["multiprocessor_count"]
        NUM_REGS = properties["max_num_regs"]
        SIZE_SMEM = properties["max_shared_mem"]
        WARP_SIZE = properties["warpSize"]
        _DEVICE_PROPERTIES = DeviceProperties(NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE)
    return _DEVICE_PROPERTIES
```

### Kernel Configuration Dataclasses

#### `KernelConfig` (Base Class)

Common configuration parameters:

```python
@dataclass
class KernelConfig:
    BLOCK_SIZE_M: int = 32
    BLOCK_SIZE_N: int = 32
    BLOCK_SIZE_K: int = 32
    num_warps: int = 4
    num_stages: int = 2
    flatten: bool = True
    permute_x: bool = False
    permute_y: bool = False
    fuse_mul_post: bool = False
    use_tma_store: bool = False

    def to_string(self, include_tuning_params: bool = False, include_tma: bool = False):
        # Returns human-readable configuration string
        s = []
        if self.permute_x:
            s.append("permute_x")
        if self.permute_y:
            s.append("permute_y")
        if include_tuning_params:
            s.append(f"BLOCK_SIZE_M={self.BLOCK_SIZE_M},"
                     f"BLOCK_SIZE_N={self.BLOCK_SIZE_N},"
                     f"BLOCK_SIZE_K={self.BLOCK_SIZE_K},"
                     f"num_warps={self.num_warps},"
                     f"num_stages={self.num_stages},"
                     f"flatten={self.flatten}")
        if include_tma:
            for f in fields(self):
                if f.name.startswith("use_tma_"):
                    if getattr(self, f.name):
                        s.append(f.name)
        return ",".join(s)
```

#### `KernelConfigForward`

Forward pass specific configuration:

```python
@dataclass
class KernelConfigForward(KernelConfig):
    use_tma_load_w: bool = False  # TMA for weight loading
    use_tma_load_x: bool = False  # TMA for activation loading
```

#### `KernelConfigBackward_dW`

Weight gradient configuration:

```python
@dataclass
class KernelConfigBackward_dW(KernelConfig):
    use_tma_load_dy: bool = False  # TMA for gradient loading
    use_tma_load_x: bool = False   # TMA for activation loading
```

#### `KernelConfigBackward_dX`

Input gradient configuration:

```python
@dataclass
class KernelConfigBackward_dX(KernelConfig):
    use_tma_load_dy: bool = False  # TMA for gradient loading
    use_tma_load_w: bool = False   # TMA for weight loading
```

### Benchmark Result Tracking

#### `KernelResult` Dataclass

Captures benchmarking results:

```python
@dataclass
class KernelResult:
    torch_time: float       # Baseline PyTorch time
    triton_time: float      # Triton kernel time
    speedup: float          # torch_time / triton_time
    kernel_config: KernelConfig

    def to_dict(self):
        return OrderedDict(
            **asdict(self.kernel_config),
            torch_time=self.torch_time,
            triton_time=self.triton_time,
            speedup=self.speedup,
        )

    @staticmethod
    def to_dataframe(results: list["KernelResult"],
                     sort_by: str = "speedup",
                     ascending: bool = False):
        df = pd.DataFrame([result.to_dict() for result in results])
        df = df.sort_values(by=sort_by, ascending=ascending)
        return df

    @staticmethod
    def to_csv(results: list["KernelResult"], filename: str = "results.csv", ...):
        # Export results to CSV

    @staticmethod
    def print_table(results: list["KernelResult"], num_results: int = 10, ...):
        # Print top results as formatted table
```

### Configuration Generation

#### `get_kernel_configs`

Generates configurations for all kernel types:

```python
def get_kernel_configs(
    BLOCK_M = DEFAULT_M_BLOCK_SIZES,    # [64, 128]
    BLOCK_N = DEFAULT_N_BLOCK_SIZES,    # [64, 128, 256]
    BLOCK_K = DEFAULT_K_BLOCK_SIZES,    # [64, 128, 256]
    num_warps = DEFAULT_NUM_WARPS,      # [4, 8]
    num_stages = DEFAULT_NUM_STAGES,    # [3, 4, 5]
    use_tma_loads = BOOLS,              # [True, False]
    fuse_permute = BOOLS,               # [True, False]
):
    kernel_configs_fwd = []
    kernel_configs_backward_dW = []
    kernel_configs_backward_dX = []

    for block_m, block_n, block_k, w, s, use_tma_load, permute in product(...):
        kernel_configs_fwd.append(
            KernelConfigForward(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=w,
                num_stages=s,
                use_tma_load_x=use_tma_load,
                use_tma_load_w=use_tma_load,
                use_tma_store=False,
                permute_x=permute,
                permute_y=permute,
            )
        )
        # Similar for backward configs...

    # Apply pruning
    kernel_configs_fwd = prune_kernel_configs_fwd(kernel_configs_fwd)
    kernel_configs_backward_dW = prune_kernel_configs_backward_dW(...)
    kernel_configs_backward_dX = prune_kernel_configs_backward_dX(...)

    return kernel_configs_fwd, kernel_configs_backward_dW, kernel_configs_backward_dX
```

### Configuration Pruning

#### `prune_kernel_configs_fwd`

Removes invalid forward configurations:

```python
def prune_kernel_configs_fwd(configs: list[KernelConfigForward]):
    pruned_configs = []
    for config in configs:
        # TMA load X incompatible with permute_x
        if config.use_tma_load_x and config.permute_x:
            continue
        # Cannot permute both input and output
        if config.permute_x and config.permute_y:
            continue
        # TMA store incompatible with permute_y
        if config.use_tma_store and config.permute_y:
            continue
        pruned_configs.append(config)
    return pruned_configs
```

#### `prune_kernel_configs_backward_dX`

Removes invalid dX backward configurations:

```python
def prune_kernel_configs_backward_dX(configs: list[KernelConfigBackward_dX]):
    pruned_configs = []
    for config in configs:
        # TMA load dY incompatible with permute_y (need scattered load)
        if config.use_tma_load_dy and config.permute_y:
            continue
        if config.permute_x and config.permute_y:
            continue
        # TMA store incompatible with permute_x (need scattered store)
        if config.use_tma_store and config.permute_x:
            continue
        pruned_configs.append(config)
    return pruned_configs
```

#### `prune_kernel_configs_backward_dW`

Removes invalid dW backward configurations:

```python
def prune_kernel_configs_backward_dW(configs: list[KernelConfigBackward_dW]):
    pruned_configs = []
    for config in configs:
        if config.use_tma_load_dy and config.permute_y:
            continue
        if config.use_tma_load_x and config.permute_x:
            continue
        if config.permute_x and config.permute_y:
            continue
        pruned_configs.append(config)
    return pruned_configs
```

### Error Handling

#### `TritonTuningContext`

Context manager for safe kernel benchmarking:

```python
class TritonTuningContext:
    def __init__(self, kernel_config: KernelConfig):
        self.kernel_config = kernel_config
        self.success = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is OutOfResources:
            name = exc_value.name
            required = exc_value.required
            limit = exc_value.limit
            print(f"Kernel config {self.kernel_config} failed: "
                  f"{name}, required: {required}, limit: {limit}")
            self.success = False
        elif exc_type is not None:
            print(f"Error running Triton grouped GEMM for "
                  f"kernel config: {self.kernel_config}: {exc_value}")
            self.success = False
        return True  # Suppress exceptions
```

Usage:

```python
with TritonTuningContext(config) as ctx:
    result = run_kernel(config)

if ctx.success:
    results.append(result)
```

## Configuration Space Summary

| Parameter | Options | Description |
|-----------|---------|-------------|
| `BLOCK_SIZE_M` | [64, 128] | Tile size along M dimension |
| `BLOCK_SIZE_N` | [64, 128, 256] | Tile size along N dimension |
| `BLOCK_SIZE_K` | [64, 128, 256] | Tile size along K (reduction) dimension |
| `num_warps` | [4, 8] | Warps per thread block |
| `num_stages` | [3, 4, 5] | Software pipelining stages |
| `use_tma_load_*` | [True, False] | TMA for loading |
| `permute_x/y` | [True, False] | Fusion of permutation |

## Typical Usage

### Manual Configuration

```python
from grouped_gemm.kernels.tuning import (
    KernelConfigForward,
    KernelConfigBackward_dX,
    KernelConfigBackward_dW,
)

# Configure for first GEMM (gate/up projection)
fwd_config = KernelConfigForward(
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=64,
    num_warps=8,
    num_stages=4,
    use_tma_load_w=True,
    use_tma_load_x=False,  # Cannot use with permute_x
    permute_x=True,
)

bwd_dX_config = KernelConfigBackward_dX(
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=64,
    num_warps=8,
    num_stages=4,
    use_tma_load_w=True,
    use_tma_store=False,  # Cannot use with permute_x
    permute_x=True,
)

bwd_dW_config = KernelConfigBackward_dW(
    BLOCK_SIZE_M=128,
    BLOCK_SIZE_N=128,
    BLOCK_SIZE_K=64,
    num_warps=8,
    num_stages=4,
    use_tma_load_dy=True,
    use_tma_load_x=False,  # Cannot use with permute_x
    permute_x=True,
)
```

### Benchmarking

```python
results = []
configs_fwd, configs_dW, configs_dX = get_kernel_configs()

for config in configs_fwd:
    with TritonTuningContext(config) as ctx:
        triton_time = benchmark_kernel(config)
        torch_time = benchmark_baseline()

    if ctx.success:
        results.append(KernelResult(
            torch_time=torch_time,
            triton_time=triton_time,
            speedup=torch_time / triton_time,
            kernel_config=config,
        ))

KernelResult.print_table(results, num_results=10)
KernelResult.to_csv(results, filename="gemm_benchmark.csv")
```

## Dependencies

- `torch`: GPU device queries
- `triton`: Runtime driver for device properties
- `pandas`: DataFrame operations for result tracking
- `dataclasses`: Configuration dataclass support
- `grouped_gemm.kernels.autotuning`: Default configuration constants

## Source File

`unsloth/kernels/moe/grouped_gemm/kernels/tuning.py` (277 lines)

## License

GNU Affero General Public License v3.0 - Copyright 2023-present the Unsloth team.
