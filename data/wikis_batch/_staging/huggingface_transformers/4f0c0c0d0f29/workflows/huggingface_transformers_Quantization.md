{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|HuggingFace Quantization|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::LLMs]], [[domain::Quantization]], [[domain::Inference_Optimization]], [[domain::Memory_Efficiency]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
End-to-end process for loading and running transformer models with quantization to reduce memory footprint and improve inference efficiency.

=== Description ===
This workflow covers the complete quantization pipeline in the Transformers library. It handles:

1. **Quantization Method Selection**: Choosing the appropriate quantization backend
2. **Configuration Setup**: Configuring quantization parameters and options
3. **Quantizer Initialization**: Creating and validating the quantizer instance
4. **Model Preparation**: Preparing the model architecture for quantized weights
5. **Weight Conversion**: Converting or loading quantized weights
6. **Runtime Optimization**: Setting up optimized kernels for inference

The library supports 20+ quantization methods including BitsAndBytes, GPTQ, AWQ, EETQ, FP8, HQQ, Quanto, and TorchAO.

=== Usage ===
Execute this workflow when you need to:
* Load large models on consumer GPUs with limited VRAM
* Run inference with INT4/INT8 quantization for speed
* Use pre-quantized models from the HuggingFace Hub
* Apply on-the-fly quantization during model loading
* Optimize inference latency with hardware-specific kernels

== Execution Steps ==

=== Step 1: Quantization Method Selection ===
[[step::Principle:huggingface_transformers_Quantization_Method_Selection]]

Select the appropriate quantization backend based on hardware, accuracy requirements, and model compatibility. Each method has different trade-offs between compression ratio, accuracy, and inference speed.

'''Available methods:'''
* BitsAndBytes (bnb): INT4/INT8, widely compatible, easy to use
* GPTQ: Weight-only, high compression, requires calibration data
* AWQ: Activation-aware, good accuracy preservation
* EETQ: Efficient INT8 for specific GPUs
* FP8: Hardware-accelerated on H100/Ada GPUs
* HQQ: Half-quadratic quantization, no calibration needed
* TorchAO: PyTorch native quantization toolkit

=== Step 2: Quantization Configuration ===
[[step::Principle:huggingface_transformers_Quantization_Config_Setup]]

Create a quantization configuration object with desired parameters. Each quantization method has its own config class with method-specific options like bit width, grouping, and computation dtype.

'''Common configuration options:'''
* Bit width: 4-bit or 8-bit (method dependent)
* Compute dtype: float16, bfloat16, float32 for matmuls
* Quantization type: NF4, FP4, INT4, INT8
* Group size: Granularity of quantization (32, 64, 128)
* Double quantization: Quantize the quantization constants

=== Step 3: Quantizer Initialization ===
[[step::Principle:huggingface_transformers_Quantizer_Initialization]]

Initialize the quantizer object that handles weight conversion and optimized kernels. The quantizer validates hardware compatibility and dependency availability before proceeding.

'''Quantizer responsibilities:'''
* Validate required dependencies are installed
* Check hardware compatibility (CUDA, ROCm, etc.)
* Verify model architecture support
* Initialize kernel backends

=== Step 4: Model Architecture Preparation ===
[[step::Principle:huggingface_transformers_Quantized_Model_Preparation]]

Prepare the model architecture to accept quantized weights. This may involve replacing linear layers with quantized equivalents or modifying the model's forward pass to use optimized kernels.

'''Preparation steps:'''
* Identify target layers for quantization (typically linear layers)
* Replace modules with quantized variants
* Configure layer-specific quantization (skip embeddings, etc.)
* Set up input/output quantization if needed

=== Step 5: Weight Conversion and Loading ===
[[step::Principle:huggingface_transformers_Quantized_Weight_Loading]]

Load pre-quantized weights or convert FP16/BF16 weights to quantized format. Pre-quantized models load directly; on-the-fly quantization converts during loading.

'''Loading modes:'''
* Pre-quantized: Load INT4/INT8 weights directly from checkpoint
* On-the-fly: Convert FP16 weights during loading
* Calibration: Run sample data through model for methods requiring calibration
* Sharded loading: Handle large quantized models with sharded weights

=== Step 6: Runtime Optimization ===
[[step::Principle:huggingface_transformers_Quantized_Runtime_Optimization]]

Configure optimized kernels and runtime settings for best inference performance. Different backends support various CUDA kernels and can be tuned for throughput vs. latency.

'''Optimizations available:'''
* Fused GEMM kernels (cuBLAS, Marlin, ExLlama)
* Flash Attention compatibility
* Batch size tuning for throughput
* Memory layout optimization
* Kernel auto-selection based on input shapes

== Execution Diagram ==
{{#mermaid:graph TD
    A[Quantization Method Selection] --> B[Quantization Configuration]
    B --> C[Quantizer Initialization]
    C --> D[Model Architecture Preparation]
    D --> E[Weight Conversion and Loading]
    E --> F[Runtime Optimization]
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Quantization_Method_Selection]]
* [[step::Principle:huggingface_transformers_Quantization_Config_Setup]]
* [[step::Principle:huggingface_transformers_Quantizer_Initialization]]
* [[step::Principle:huggingface_transformers_Quantized_Model_Preparation]]
* [[step::Principle:huggingface_transformers_Quantized_Weight_Loading]]
* [[step::Principle:huggingface_transformers_Quantized_Runtime_Optimization]]
