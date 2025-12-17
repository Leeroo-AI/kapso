# RLHF with Online Quantization Example

**Source:** `examples/offline_inference/rlhf_online_quant.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 162

## Overview

This example demonstrates how to combine RLHF training with online quantization using Float8 (FP8) precision. It extends the basic RLHF pattern with TorchAO quantization, enabling memory-efficient inference during rollout generation while maintaining the ability to synchronize weights from full-precision training.

## Implementation Pattern

### Architecture Design

The implementation follows the same separated GPU pattern as the basic RLHF example, but adds quantization to the inference engine:

**Training Process (GPU 0):**
- Hosts the model in full precision (FP32/FP16)
- Performs gradient updates normally
- Source of truth for model weights

**Inference Engine (GPUs 1-2):**
- Runs vLLM with Float8 quantization
- Uses tensor parallelism across 2 GPUs
- Accepts weight updates and requantizes on-the-fly

### Key Innovation

The example shows how to configure quantization for RLHF rollout generation, reducing memory usage without modifying the training pipeline. This enables:
- Larger batch sizes during rollout
- Support for larger models on limited GPU memory
- Faster inference through reduced memory bandwidth

## Technical Implementation

### 1. Quantization Configuration

```python
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
)
from torchao.core.config import config_to_dict

# Generate torchao quantization config for RL rollout
config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())

# Serialize to JSON for passing to vLLM
json_str = json.dumps(config_to_dict(config))
```

**Quantization Settings:**
- `Float8DynamicActivationFloat8WeightConfig`: Both activations and weights in FP8
- `PerRow` granularity: Quantization parameters computed per matrix row
- Dynamic activations: Quantization scales computed at runtime

### 2. Passing Configuration to vLLM

```python
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model="facebook/opt-125m",
    hf_overrides={"quantization_config_dict_json": json_str},
    enforce_eager=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)
```

**Configuration Method:**
The quantization configuration is passed via `hf_overrides` with the key `quantization_config_dict_json`. This approach:
- Avoids passing Python objects across Ray actors
- Enables serialization/deserialization of complex configs
- Future versions may support loading from config files (see PR #23014)

### 3. Complete Workflow

The example follows the same workflow as the base RLHF example:

```python
# 1. Generate text from prompts with quantized inference
outputs = ray.get(llm.generate.remote(prompts, sampling_params))

# 2. Simulate training step (zero weights for demonstration)
for name, p in train_model.named_parameters():
    p.data.zero_()

# 3. Synchronize updated weights to quantized inference engine
for name, p in train_model.named_parameters():
    dtype_name = str(p.dtype).split(".")[-1]
    handle = llm.collective_rpc.remote(
        "update_weight", args=(name, dtype_name, p.shape)
    )
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)

# 4. Verify weight updates applied correctly
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))

# 5. Generate text with updated quantized model
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
```

## Float8 Quantization Details

### Why Float8?

**Memory Benefits:**
- FP8 uses 1 byte per parameter vs. 2 bytes (FP16) or 4 bytes (FP32)
- Reduces memory footprint by 50% (vs FP16) or 75% (vs FP32)
- Enables larger batch sizes or larger models

**Computational Benefits:**
- Modern GPUs (H100, H200) have dedicated FP8 Tensor Cores
- Higher throughput for matrix operations
- Reduced memory bandwidth requirements

### Dynamic vs. Static Quantization

**Dynamic Activations:**
```python
Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
```

This configuration:
- Quantizes weights statically (once during initialization or update)
- Quantizes activations dynamically (each forward pass)
- Computes scaling factors on-the-fly for activations
- Balances accuracy and performance

**Per-Row Granularity:**
Scaling factors are computed per row of weight matrices, providing fine-grained quantization that maintains accuracy better than per-tensor approaches.

## Weight Update Handling

### Requantization Process

When weights are updated from the training process:

1. **Full-Precision Broadcast:** Training sends FP32/FP16 weights via NCCL
2. **Worker Reception:** Inference worker receives full-precision weights
3. **Automatic Requantization:** vLLM automatically requantizes to FP8
4. **Model Update:** Quantized weights loaded into the model

This process is handled transparently by the `WorkerExtension.update_weight()` method in conjunction with vLLM's quantization layers.

## Configuration Serialization

### Current Approach: JSON Serialization

```python
from torchao.core.config import config_to_dict
json_str = json.dumps(config_to_dict(config))
```

The configuration is converted to a dictionary and serialized to JSON for passing across process boundaries.

### Future Approach: Config Files

The example notes that PR #23014 introduces support for loading quantization configs from files:

```python
# Future syntax (not yet available)
llm = LLM(
    model="facebook/opt-125m",
    quantization_config_file="path/to/config.json"
)
```

This will simplify configuration management for production deployments.

## Usage Requirements

### System Requirements
- GPUs with FP8 support (H100/H200) for optimal performance
- Fallback to simulated FP8 on older GPUs (slower)
- Minimum 3 GPUs (1 for training, 2 for inference)

### Dependencies
- **TorchAO:** Required for quantization configuration
- **Ray:** For distributed execution
- **PyTorch:** 2.0+ with FP8 support
- **vLLM:** Built with quantization support

### Installation

```bash
# Install TorchAO
pip install torchao

# Ensure vLLM has quantization support
pip install vllm  # or build from source
```

## Performance Considerations

### Memory Savings

**Example Model (OPT-125M):**
- Full precision (FP32): ~500 MB
- Half precision (FP16): ~250 MB
- Float8: ~125 MB

**For Larger Models (e.g., Llama-70B):**
- FP16: ~140 GB
- FP8: ~70 GB (fits on 2x A100 80GB instead of 4x)

### Throughput Impact

**On H100/H200 (native FP8):**
- 1.5-2x higher throughput vs FP16
- Lower latency due to reduced memory transfers

**On A100/V100 (simulated FP8):**
- Slight overhead from quantization operations
- Still benefits from reduced memory bandwidth

### Accuracy Considerations

**Quantization Error:**
Float8 quantization introduces small numerical errors. For RLHF:
- Rollout generation: Minor impact, as sampling adds randomness
- Policy gradients: Computed on full-precision model
- Overall training: Minimal impact on convergence

**Best Practices:**
- Use dynamic activation quantization for better accuracy
- Monitor reward metrics to detect quantization issues
- Consider per-row or per-channel granularity over per-tensor

## Production Deployment Patterns

### Hybrid Precision Strategy

**Optimal Configuration:**
- Training: Full precision (FP32) or mixed precision (FP16 with AMP)
- Inference: Float8 quantization for rollouts
- Evaluation: Full precision for final metrics

### Scaling Considerations

**Vertical Scaling (Larger Models):**
Quantization enables fitting larger models on available GPUs:
```python
# Without quantization: Requires 4x A100 80GB
# With FP8: Fits on 2x A100 80GB
llm = MyLLM(
    model="meta-llama/Llama-2-70b-hf",
    hf_overrides={"quantization_config_dict_json": json_str},
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
)
```

**Horizontal Scaling (More Replicas):**
Run multiple quantized inference engines with the same GPU budget:
```python
# With quantization: 4x more replicas on same GPUs
for i in range(4):
    llm = create_quantized_inference_engine(
        bundle_indices=[2*i, 2*i+1]
    )
```

## Integration with Training Pipeline

### RLHF Training Loop

```python
for iteration in range(num_iterations):
    # 1. Generate rollouts with quantized inference
    rollouts = ray.get(llm.generate.remote(prompts, sampling_params))

    # 2. Compute rewards
    rewards = reward_model(rollouts)

    # 3. Update policy with PPO
    policy_optimizer.step(rollouts, rewards)

    # 4. Synchronize updated weights to inference
    synchronize_weights(train_model, llm, model_update_group)

    # 5. Optional: Evaluate with full precision
    if iteration % eval_frequency == 0:
        evaluate_full_precision(train_model)
```

### Weight Update Frequency

**High-Frequency Updates:**
For algorithms like PPO that update frequently:
- Quantization overhead amortized over many inferences
- Fast NCCL broadcast critical for maintaining throughput

**Low-Frequency Updates:**
For algorithms with larger batch sizes:
- Requantization overhead negligible
- Focus on maximizing inference throughput between updates

## Troubleshooting

### Common Issues

**Quantization Not Applied:**
- Verify TorchAO is installed: `pip list | grep torchao`
- Check vLLM was built with quantization support
- Ensure JSON serialization is correct

**Accuracy Degradation:**
- Try per-channel instead of per-tensor granularity
- Use dynamic activation quantization
- Verify training model isn't also quantized

**Memory Issues:**
- Reduce `gpu_memory_utilization` parameter
- Check for memory leaks during weight updates
- Monitor with `nvidia-smi` during execution

## Related Examples

- `rlhf.py`: Base RLHF example without quantization
- `rlhf_colocate.py`: Co-located training/inference pattern
- `rlhf_utils.py`: Shared utilities including WorkerExtension

## References

- **TorchAO Quantization:** [GitHub](https://github.com/pytorch/ao)
- **Float8 Documentation:** PyTorch FP8 recipes
- **vLLM Quantization:** [PR #23014](https://github.com/vllm-project/vllm/pull/23014)
- **RLHF Background:** Proximal Policy Optimization (PPO) algorithm
