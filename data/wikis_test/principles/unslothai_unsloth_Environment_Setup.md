{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Unsloth Installation|https://docs.unsloth.ai/get-started/installation]]
* [[source::Doc|PyTorch CUDA Setup|https://pytorch.org/get-started/locally/]]
* [[source::Doc|BitsAndBytes Installation|https://github.com/TimDettmers/bitsandbytes]]
|-
! Domains
| [[domain::DevOps]], [[domain::Deep_Learning]], [[domain::GPU_Computing]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Process of configuring hardware detection, dependency installation, and runtime optimization for efficient LLM fine-tuning on various GPU architectures.

=== Description ===
Environment setup for LLM fine-tuning involves configuring the software stack to efficiently utilize available hardware. Key components include:

1. **GPU Detection**: Identify CUDA compute capability, available VRAM, and supported features
2. **Precision Selection**: Choose appropriate dtype (float16, bfloat16) based on GPU architecture
3. **Memory Optimization**: Configure gradient checkpointing, attention backends, and memory allocators
4. **Dependency Management**: Install compatible versions of PyTorch, BitsAndBytes, Triton, and PEFT

Hardware-specific optimizations:
- **Ampere+ (A100, RTX 30xx/40xx)**: BF16 compute, TF32 for matmuls, Flash Attention 2
- **Turing (RTX 20xx, T4)**: FP16 compute, Flash Attention 1
- **Volta (V100)**: FP16 with slower accumulation
- **AMD (ROCm)**: HIP-based kernels, limited Flash Attention support

The setup process also handles common issues like CUDA version mismatches, BitsAndBytes compilation for specific architectures, and Triton JIT compilation caching.

=== Usage ===
Perform environment setup when:
- Installing Unsloth for the first time
- Moving to a new GPU or cloud instance
- Troubleshooting training crashes or performance issues
- Verifying hardware utilization

Key verification steps:
- Confirm CUDA toolkit version matches PyTorch build
- Verify BitsAndBytes can load CUDA extensions
- Check available GPU memory aligns with expectations
- Test basic model loading before full training

== Theoretical Basis ==
Environment configuration optimizes the memory hierarchy and compute precision for transformer training.

'''GPU Architecture Detection:'''
<syntaxhighlight lang="python">
# Pseudo-code for hardware detection
def detect_gpu_capabilities():
    """
    Detect GPU features to configure optimal settings.
    """
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "dtype": torch.float32}

    # Get compute capability
    major, minor = torch.cuda.get_device_capability()
    compute_cap = major * 10 + minor

    config = {
        "device": "cuda",
        "compute_capability": compute_cap,
        "device_name": torch.cuda.get_device_name(),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }

    # Determine optimal dtype
    if compute_cap >= 80:  # Ampere+
        config["dtype"] = torch.bfloat16
        config["supports_flash_attn_2"] = True
        config["supports_tf32"] = True
    elif compute_cap >= 75:  # Turing
        config["dtype"] = torch.float16
        config["supports_flash_attn_2"] = True
        config["supports_tf32"] = False
    else:  # Older
        config["dtype"] = torch.float16
        config["supports_flash_attn_2"] = False
        config["supports_tf32"] = False

    return config
</syntaxhighlight>

'''Memory Configuration:'''
<syntaxhighlight lang="python">
# Pseudo-code for memory optimization setup
def configure_memory_optimization(vram_gb, model_size_b):
    """
    Configure memory settings based on available resources.
    """
    config = {}

    # Estimate memory requirements
    # 4-bit model: ~0.5 GB per billion parameters
    # FP16 model: ~2 GB per billion parameters
    # Training overhead: ~2-3x model size

    model_4bit_gb = model_size_b * 0.5
    training_overhead = 2.5

    if vram_gb < model_4bit_gb * training_overhead:
        # Limited memory: aggressive optimization
        config["load_in_4bit"] = True
        config["gradient_checkpointing"] = "unsloth"  # 30% savings
        config["optim"] = "adamw_8bit"  # 8-bit optimizer
        config["per_device_batch_size"] = 1
        config["gradient_accumulation"] = 8
    else:
        # Sufficient memory: balanced settings
        config["load_in_4bit"] = True
        config["gradient_checkpointing"] = "unsloth"
        config["optim"] = "adamw_torch_fused"
        config["per_device_batch_size"] = 2
        config["gradient_accumulation"] = 4

    return config
</syntaxhighlight>

'''Precision and Backend Selection:'''
<syntaxhighlight lang="text">
GPU Architecture Decision Tree:
┌─────────────────────────────────────────────────────┐
│ Compute Capability >= 8.0 (Ampere: A100, RTX 3090)  │
│   ├─ Use: torch.bfloat16                            │
│   ├─ Enable: TF32 for matmuls                       │
│   ├─ Attention: Flash Attention 2                    │
│   └─ Memory: Most efficient                          │
├─────────────────────────────────────────────────────┤
│ Compute Capability >= 7.5 (Turing: T4, RTX 2080)    │
│   ├─ Use: torch.float16                              │
│   ├─ Attention: Flash Attention 2 (fallback FA1)    │
│   └─ Memory: Good efficiency                         │
├─────────────────────────────────────────────────────┤
│ Compute Capability >= 7.0 (Volta: V100)             │
│   ├─ Use: torch.float16                              │
│   ├─ Attention: SDPA or manual                       │
│   └─ Memory: Moderate efficiency                     │
└─────────────────────────────────────────────────────┘
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Tips and Tricks ===
