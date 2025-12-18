{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Documentation|https://huggingface.co/docs/transformers/main_classes/quantization]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Software_Engineering]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete validation method for checking quantization environment compatibility provided by HuggingFace Transformers.

=== Description ===
HfQuantizer.validate_environment implements the Quantization_Validation principle as an abstract base method that each quantizer backend overrides with specific validation logic. The base implementation provides a no-op default, while concrete quantizers (Bnb4BitQuantizer, GptqQuantizer, etc.) implement checks for their specific requirements. This design allows the model loading pipeline to call validate_environment uniformly across all quantization backends while each backend performs appropriate checks.

=== Usage ===
This method is called automatically by the model loading pipeline after quantizer selection. Override it when implementing a new quantizer backend to check:
* Required library availability and versions
* Hardware compatibility (CUDA version, compute capability)
* Device map strategy compatibility
* Parameter conflicts with other loading options
* Memory availability for quantization overhead

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/base.py
* '''Lines:''' 150-157

=== Signature ===
<syntaxhighlight lang="python">
class HfQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers
    models for inference and/or quantization.
    """

    def validate_environment(self, *args, **kwargs):
        """
        This method is used to potentially check for potential conflicts with arguments that are
        passed in `from_pretrained`. You need to define it for all future quantizers that are
        integrated with transformers. If no explicit check are needed, simply return nothing.

        Args:
            *args: Positional arguments passed from from_pretrained.
            **kwargs: Keyword arguments passed from from_pretrained. Common kwargs:
                - device_map (dict or str): Device placement strategy
                - torch_dtype (torch.dtype): Target dtype for model
                - low_cpu_mem_usage (bool): Memory optimization flag
                - weights_only (bool): Safe loading mode

        Returns:
            None: Method validates and raises exceptions on failure.

        Raises:
            ImportError: If required quantization library is not available.
            RuntimeError: If hardware requirements are not met.
            ValueError: If configuration conflicts with other parameters.
        """
        return
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| *args || tuple || No || Positional arguments from from_pretrained call
|-
| device_map || dict or str || No || Device placement strategy (e.g., "auto", "balanced")
|-
| torch_dtype || torch.dtype || No || Target dtype for non-quantized modules
|-
| low_cpu_mem_usage || bool || No || Whether to use memory-efficient loading
|-
| weights_only || bool || No || Whether to load only weights (safe loading)
|-
| **kwargs || dict || No || Additional from_pretrained arguments
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || NoneType || Method raises exception on validation failure, returns None on success
|}

== Usage Examples ==

=== Base Class No-Op Implementation ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer

# Base implementation does nothing
class MyQuantizer(HfQuantizer):
    # If validation not needed, inherit default
    pass

quantizer = MyQuantizer(config)
quantizer.validate_environment(device_map="auto")  # Returns immediately
</syntaxhighlight>

=== BitsAndBytes 4-bit Validation (Example) ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer
from transformers.utils import is_torch_available, is_bitsandbytes_available

class Bnb4BitQuantizer(HfQuantizer):
    def validate_environment(self, *args, device_map=None, **kwargs):
        # Check library availability
        if not is_bitsandbytes_available():
            raise ImportError(
                "Using `load_in_4bit=True` requires bitsandbytes library. "
                "Install it with: pip install bitsandbytes"
            )

        # Check CUDA availability
        if not is_torch_available() or not torch.cuda.is_available():
            raise RuntimeError(
                "BitsAndBytes quantization requires CUDA. "
                "No CUDA-capable device detected."
            )

        # Check compute capability
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] < 7 or (capability[0] == 7 and capability[1] < 5):
                raise RuntimeError(
                    f"BitsAndBytes quantization requires compute capability >= 7.5. "
                    f"Your device has capability {capability[0]}.{capability[1]}."
                )

        # Check device_map compatibility
        if device_map is None:
            raise ValueError(
                "Using `load_in_4bit=True` requires `device_map` argument. "
                "Pass device_map='auto' or a custom device map."
            )

        # Check for conflicting dtype
        if "torch_dtype" in kwargs and kwargs["torch_dtype"] is not None:
            import warnings
            warnings.warn(
                "torch_dtype is ignored when load_in_4bit=True. "
                "Computation dtype is controlled by bnb_4bit_compute_dtype."
            )

# Usage in model loading pipeline
try:
    quantizer = Bnb4BitQuantizer(config)
    quantizer.validate_environment(device_map="auto", torch_dtype=torch.float16)
except ImportError as e:
    print(f"Missing dependency: {e}")
except RuntimeError as e:
    print(f"Hardware incompatible: {e}")
</syntaxhighlight>

=== GPTQ Validation (Example) ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer

class GptqQuantizer(HfQuantizer):
    def validate_environment(self, *args, device_map=None, **kwargs):
        # Check auto-gptq availability
        try:
            import auto_gptq
        except ImportError:
            raise ImportError(
                "Loading GPTQ models requires auto-gptq library. "
                "Install it with: pip install auto-gptq"
            )

        # GPTQ works on CPU or CUDA
        # Less restrictive than BitsAndBytes

        # Check for pre-quantized model
        if not self.pre_quantized:
            raise ValueError(
                "GPTQ quantization requires pre-quantized model weights. "
                "Cannot quantize on-the-fly. Set pre_quantized=True."
            )

        # Validate group_size compatibility
        if hasattr(self.quantization_config, "group_size"):
            if self.quantization_config.group_size <= 0:
                raise ValueError(
                    f"Invalid group_size: {self.quantization_config.group_size}. "
                    "Must be positive integer."
                )

# Usage
try:
    quantizer = GptqQuantizer(gptq_config, pre_quantized=True)
    quantizer.validate_environment(device_map="auto")
    # Validation passed
except Exception as e:
    print(f"Validation failed: {e}")
</syntaxhighlight>

=== Custom Quantizer with Multi-Step Validation ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer
import torch

class CustomQuantizer(HfQuantizer):
    def validate_environment(self, *args, device_map=None, **kwargs):
        # Step 1: Check dependencies
        self._check_dependencies()

        # Step 2: Validate hardware
        self._check_hardware()

        # Step 3: Validate configuration
        self._check_config_compatibility(device_map, **kwargs)

    def _check_dependencies(self):
        try:
            import custom_quant_lib
            if custom_quant_lib.__version__ < "1.0.0":
                raise ImportError(
                    f"custom_quant_lib version {custom_quant_lib.__version__} "
                    "is too old. Please upgrade: pip install --upgrade custom_quant_lib"
                )
        except ImportError:
            raise ImportError(
                "Custom quantization requires custom_quant_lib. "
                "Install it from: https://github.com/example/custom_quant_lib"
            )

    def _check_hardware(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Custom quantization requires CUDA")

        # Check minimum GPU memory
        min_memory_gb = 8
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if free_memory < min_memory_gb:
                raise RuntimeError(
                    f"Custom quantization requires at least {min_memory_gb}GB GPU memory. "
                    f"Available: {free_memory:.1f}GB"
                )

    def _check_config_compatibility(self, device_map, **kwargs):
        # Check device_map
        if device_map not in ["auto", "cuda"]:
            raise ValueError(
                f"Custom quantization requires device_map='auto' or 'cuda'. "
                f"Got: {device_map}"
            )

        # Check for conflicting flags
        if kwargs.get("low_cpu_mem_usage", False):
            raise ValueError(
                "Custom quantization is incompatible with low_cpu_mem_usage=True. "
                "Set low_cpu_mem_usage=False."
            )
</syntaxhighlight>

=== Integration in Model Loading ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.quantizers.auto import AutoHfQuantizer

def from_pretrained_with_validation(model_name, quantization_config, **kwargs):
    """Simplified version of from_pretrained showing validation step"""

    # Step 1: Create quantizer
    quantizer = AutoHfQuantizer.from_config(quantization_config)

    # Step 2: Validate environment BEFORE loading any weights
    try:
        quantizer.validate_environment(
            device_map=kwargs.get("device_map"),
            torch_dtype=kwargs.get("torch_dtype"),
            **kwargs
        )
    except Exception as e:
        print(f"Validation failed: {e}")
        print("Aborting model load to save time and resources.")
        raise

    # Step 3: Proceed with model loading only if validation passed
    print("Validation passed. Loading model...")
    # ... continue with model initialization and weight loading ...

# Usage
config = BitsAndBytesConfig(load_in_4bit=True)
try:
    model = from_pretrained_with_validation(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=config,
        device_map="auto"
    )
except ImportError:
    print("Install bitsandbytes to use 4-bit quantization")
except RuntimeError:
    print("Your hardware doesn't support 4-bit quantization")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Quantization_Validation]]

=== Called By ===
* [[called_by::Implementation:huggingface_transformers_AutoHfQuantizer_dispatch]]

=== Precedes ===
* [[precedes::Implementation:huggingface_transformers_Quantizer_preprocess]]
