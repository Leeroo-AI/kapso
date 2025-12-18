{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/main_classes/quantization]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Software_Engineering]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Dynamically dispatch to the appropriate quantizer implementation based on configuration metadata using a registry pattern.

=== Description ===
Quantizer selection addresses the challenge of supporting multiple quantization backends (bitsandbytes, GPTQ, AWQ, GGUF, etc.) through a unified interface. This principle establishes a factory pattern where quantization configurations carry metadata identifying their target backend, and a registry maps these identifiers to concrete quantizer implementations. The selection happens at model loading time, before any weights are touched, ensuring the correct quantization strategy is applied consistently.

This separation enables the quantization framework to be extended with new methods without modifying the model loading pipeline. Each quantizer backend can have its own dependencies, initialization requirements, and validation logic, while the selection mechanism remains constant.

=== Usage ===
Apply this principle when you need to:
* Support multiple quantization backends through a single API entry point
* Load models that were quantized with different methods
* Add new quantization schemes without breaking existing code
* Deserialize quantization configs from model checkpoints and instantiate the correct handler
* Validate that quantization method requirements are met before loading

== Theoretical Basis ==

=== Factory Pattern with Registry ===

<pre>
Registry mapping:
QUANTIZER_REGISTRY = {
    "bitsandbytes_4bit": BnbQuantizer,
    "bitsandbytes_8bit": Bnb8BitQuantizer,
    "gptq": GptqQuantizer,
    "awq": AwqQuantizer,
    "gguf": GgufQuantizer,
    ...
}

Selection algorithm:
1. Extract method identifier from config: method = config.quant_method
2. Apply method-specific transformations (e.g., "bitsandbytes" + "_4bit")
3. Look up quantizer class: quantizer_cls = REGISTRY[method]
4. Instantiate with config: quantizer = quantizer_cls(config, **kwargs)
5. Return quantizer instance for use in loading pipeline
</pre>

=== Configuration Polymorphism ===

All quantization configs share a common interface:

<pre>
class QuantizationConfigMixin:
    quant_method: str  # Identifies the quantization backend

    # Serialization protocol
    def to_dict() -> dict
    def to_json_file(path: str)

    @classmethod
    def from_dict(data: dict) -> QuantizationConfig

Concrete implementations:
- BitsAndBytesConfig: quant_method = "bitsandbytes"
- GPTQConfig: quant_method = "gptq"
- AwqConfig: quant_method = "awq"

Selection leverages polymorphism:
config = QuantizationConfig.from_pretrained(model_path)
quantizer = AutoQuantizer.from_config(config)  # Dispatches to correct subclass
</pre>

=== Special Case Handling ===

Some backends require additional disambiguation:

<pre>
Example: bitsandbytes supports both 4-bit and 8-bit
- Base method: "bitsandbytes"
- Needs suffix: "_4bit" or "_8bit"
- Determined by: config.load_in_4bit or config.load_in_8bit flags

Selection logic:
if method == "bitsandbytes":
    if config.load_in_8bit:
        method = "bitsandbytes_8bit"
    elif config.load_in_4bit:
        method = "bitsandbytes_4bit"
    else:
        raise ValueError("Must specify load_in_4bit or load_in_8bit")
</pre>

=== Error Handling ===

<pre>
Validation steps:
1. Check method in registry:
   if method not in REGISTRY:
       raise ValueError(f"Unknown method: {method}")

2. Verify backend dependencies:
   if method == "bitsandbytes" and not is_bitsandbytes_available():
       raise ImportError("pip install bitsandbytes")

3. Validate config compatibility:
   quantizer.validate_environment(device_map, kwargs)
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_AutoHfQuantizer_dispatch]]
