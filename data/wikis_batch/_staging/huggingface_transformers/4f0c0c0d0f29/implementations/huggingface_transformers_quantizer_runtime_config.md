{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]], [[domain::GPU Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Backend-specific kernel configuration and runtime optimization settings for efficient quantized inference.

=== Description ===

Runtime configuration encompasses all kernel-specific settings that control how quantized operations execute. This includes:

* Backend selection (marlin, exllama, CUDA, Triton)
* Kernel-specific parameters (block sizes, tile shapes)
* Hardware-specific optimizations
* Fallback strategies

Configuration is typically embedded in quantization configs (e.g., GPTQConfig.backend, AwqConfig.backend) and affects which computational paths are taken during inference.

=== Usage ===

Specify backend in quantization config when loading models. The system selects optimal kernels based on available hardware and installed libraries. Advanced users can tune kernel parameters for specific workloads.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''Files:'''
  * src/transformers/utils/quantization_config.py (config classes)
  * src/transformers/quantizers/quantizer_gptq.py (GPTQ backends)
  * src/transformers/quantizers/quantizer_awq.py (AWQ backends)

=== Backend Configuration Examples ===

<syntaxhighlight lang="python">
# GPTQ Config
class GPTQConfig:
    backend: str = "auto"  # auto, marlin, exllama_v2, exllama_v1, gemm, etc.
    max_input_length: int = None  # For exllama backend

# AWQ Config
class AwqConfig:
    backend: AwqBackend = AwqBackend.AUTO
    # Options: AUTO, MARLIN, EXLLAMA_V2, GEMM, etc.

# TorchAO Config
class TorchAoConfig:
    quant_type: str = "int4_weight_only"
    quant_type_kwargs: dict = {"layout": Int4CPULayout()}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import GPTQConfig, AwqConfig, TorchAoConfig
from transformers.utils.quantization_config import AwqBackend
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| backend || str or Enum || No || Kernel backend selection
|-
| max_input_length || int || No || Maximum sequence length for buffers
|-
| kernel_params || dict || No || Backend-specific tuning parameters
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| runtime_config || dict || Configured runtime settings for kernels
|}

== Usage Examples ==

=== GPTQ with Marlin Backend ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, GPTQConfig

# Marlin backend: optimized for A100/H100
config = GPTQConfig(
    bits=4,
    group_size=128,
    backend="marlin",  # Use Marlin kernels
)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    quantization_config=config,
    device_map="auto",
)

# Marlin kernels provide:
# - 3-4x speedup over baseline
# - Fused dequant-matmul
# - Tensor core utilization
# - Optimized for group-wise quantization

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")

# Fast inference with Marlin kernels
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
</syntaxhighlight>

=== AWQ with Auto Backend Selection ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AwqConfig, AwqBackend

# Auto selects best available backend
config = AwqConfig(
    bits=4,
    group_size=128,
    backend=AwqBackend.AUTO,  # Automatically choose best backend
)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ",
    quantization_config=config,
    device_map="auto",
)

# Auto selection priority:
# 1. MARLIN (if available and compatible)
# 2. EXLLAMA_V2 (if available)
# 3. GEMM (fallback)

print(f"Selected backend: {model.config.quantization_config.backend}")
</syntaxhighlight>

=== Exllama Backend with Sequence Length ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, GPTQConfig

# Exllama requires max_input_length for buffer allocation
config = GPTQConfig(
    bits=4,
    group_size=128,
    backend="exllama_v2",
    max_input_length=4096,  # Pre-allocate buffers for 4k context
)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-13B-GPTQ",
    quantization_config=config,
    device_map="auto",
)

# Exllama optimizations:
# - Flash attention integration
# - Custom CUDA kernels
# - Good for consumer GPUs (RTX 3090, 4090)
# - Efficient memory management
</syntaxhighlight>

=== TorchAO with CPU Layout ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, TorchAoConfig
import torch

# CPU-optimized quantization
if torch.cuda.is_available():
    layout_kwargs = {}
else:
    from torchao.dtypes import Int4CPULayout
    layout_kwargs = {"layout": Int4CPULayout()}

config = TorchAoConfig(
    quant_type="int4_weight_only",
    quant_type_kwargs=layout_kwargs,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config,
    device_map="cpu",
)

# CPU-optimized kernels for inference without GPU
</syntaxhighlight>

=== Backend Fallback Strategy ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, GPTQConfig

# Try multiple backends with fallback
backends_to_try = ["marlin", "exllama_v2", "gemm"]

for backend in backends_to_try:
    try:
        config = GPTQConfig(
            bits=4,
            group_size=128,
            backend=backend,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7B-GPTQ",
            quantization_config=config,
            device_map="auto",
        )

        print(f"Successfully loaded with {backend} backend")
        break
    except Exception as e:
        print(f"Backend {backend} failed: {e}")
        continue
</syntaxhighlight>

=== Kernel Tuning Parameters ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, HiggsConfig

# Higgs with kernel tuning
config = HiggsConfig(
    bits=4,
    group_size=256,
    hadamard_size=512,
    tune_metadata={
        "model.layers.0.self_attn.q_proj": {
            "block_shape": [128, 128],
            "num_warps": 4,
        },
        # More layer-specific tuning...
    }
)

model = AutoModelForCausalLM.from_pretrained(
    "model-with-higgs",
    quantization_config=config,
    device_map="auto",
)

# tune_metadata specifies kernel launch parameters per layer
</syntaxhighlight>

== Backend Comparison ==

{| class="wikitable"
|-
! Backend !! Speed !! Memory !! Compatibility !! Best For
|-
| Marlin || Excellent || Low || A100/H100 || Production inference
|-
| Exllama V2 || Very Good || Low || Most CUDA GPUs || Consumer GPUs
|-
| Exllama V1 || Good || Low || Most CUDA GPUs || Legacy models
|-
| GEMM || Good || Medium || All GPUs || Fallback/compatibility
|-
| Triton || Variable || Low || Modern GPUs || Research/flexibility
|-
| cuBLAS || Good || Medium || All NVIDIA GPUs || Reliable baseline
|-
| CPU Layouts || Fair || Low || Any CPU || CPU-only inference
|}

== Performance Tuning ==

=== Batch Size Effects ===
<syntaxhighlight lang="python">
import torch
import time

model = ...  # Loaded with specific backend

# Test different batch sizes
for batch_size in [1, 2, 4, 8, 16, 32]:
    inputs = torch.randn(batch_size, 128, device="cuda")

    # Warmup
    for _ in range(10):
        model(inputs)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        model(inputs)
    torch.cuda.synchronize()
    end = time.time()

    throughput = (batch_size * 100) / (end - start)
    print(f"Batch {batch_size}: {throughput:.2f} samples/sec")
</syntaxhighlight>

=== Backend Selection Heuristics ===

'''For A100/H100 GPUs:'''
* GPTQ → Marlin backend
* AWQ → Marlin or GEMM_TRITON
* FP8 → Native FP8 kernels

'''For RTX 3090/4090:'''
* GPTQ → Exllama V2
* AWQ → Exllama V2
* BnB → cuBLAS INT8

'''For CPU:'''
* TorchAO with Int4CPULayout
* Quanto with CPU backend

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantized_Runtime_Optimization]]

=== Requires ===
* [[requires::Implementation:huggingface_transformers_quantizer_postprocess_model]]

=== Depends On ===
