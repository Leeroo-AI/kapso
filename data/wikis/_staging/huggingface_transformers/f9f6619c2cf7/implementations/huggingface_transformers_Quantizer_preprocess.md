{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Quantization Documentation|https://huggingface.co/docs/transformers/main_classes/quantization]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete preprocessing method for preparing model architecture for quantized weight loading provided by HuggingFace Transformers.

=== Description ===
HfQuantizer.preprocess_model implements the Weight_Quantization principle by modifying the model skeleton on the meta device before weights are loaded. It sets quantization flags, attaches configuration metadata, optionally converts module types for pre-quantized formats, and delegates to backend-specific preprocessing logic. The method ensures that the model structure is ready to receive quantized weights in the correct format, whether through on-the-fly quantization or direct loading of pre-quantized tensors.

=== Usage ===
This method is called automatically by the model loading pipeline after environment validation. Override _process_model_before_weight_loading in custom quantizers to implement backend-specific preprocessing such as:
* Replacing Linear modules with custom quantized layer types
* Setting module-level quantization parameters
* Adjusting dtype for specific modules
* Modifying device placement strategies
* Registering custom weight loading hooks

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/quantizers/base.py
* '''Lines:''' 169-186

=== Signature ===
<syntaxhighlight lang="python">
class HfQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer.
    """

    def preprocess_model(self, model: "PreTrainedModel", dtype=None, **kwargs):
        """
        Setting model attributes and/or converting model before weights loading. At this point
        the model should be initialized on the meta device so you can freely manipulate the skeleton
        of the model in order to replace modules in-place. Make sure to override the abstract method
        `_process_model_before_weight_loading`.

        Args:
            model (~transformers.PreTrainedModel):
                The model to quantize. Should be on meta device.
            dtype (torch.dtype, optional):
                Target dtype for non-quantized modules.
            **kwargs:
                Additional keyword arguments passed to `_process_model_before_weight_loading`.

        Returns:
            None: Model is modified in-place.
        """
        model.is_quantized = True
        model.quantization_method = self.quantization_config.quant_method
        if self.pre_quantized:
            self._convert_model_for_quantization(model)
        self._process_model_before_weight_loading(model, **kwargs)

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        """
        Backend-specific preprocessing. Override this method in concrete quantizers.

        Args:
            model (~transformers.PreTrainedModel): The model to preprocess.
            **kwargs: Backend-specific arguments.

        Returns:
            PreTrainedModel: The preprocessed model.
        """
        return model

    def _convert_model_for_quantization(self, model):
        """
        Convert modules for pre-quantized weight formats (e.g., GPTQ, AWQ).
        Replaces standard modules with quantization-specific implementations.

        Args:
            model (~transformers.PreTrainedModel): The model to convert.
        """
        from accelerate import init_empty_weights

        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name in MODULES_TO_PATCH_FOR_QUANTIZATION and (
                self.quantization_config.quant_method
                in MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["quantization_methods"]
            ):
                with init_empty_weights():
                    parent_module, name = get_module_from_name(model, name)
                    parent_module._modules[name] = MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["module_name"](
                        model.config.get_text_config()
                    )
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
| model || PreTrainedModel || Yes || Model on meta device to preprocess
|-
| dtype || torch.dtype || No || Target dtype for non-quantized modules
|-
| **kwargs || dict || No || Backend-specific preprocessing arguments
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || NoneType || Model is modified in-place with quantization attributes set
|}

== Usage Examples ==

=== Base Preprocessing Flow ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.quantizers.auto import AutoHfQuantizer
import torch

# Initialize model on meta device
with torch.device("meta"):
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_config(config)

# Create quantizer
quant_config = BitsAndBytesConfig(load_in_4bit=True)
quantizer = AutoHfQuantizer.from_config(quant_config)

# Preprocess model (sets attributes, prepares for quantization)
quantizer.preprocess_model(model, dtype=torch.float16)

# Check quantization flags
print(model.is_quantized)  # True
print(model.quantization_method)  # "bitsandbytes"

# Model is now ready for weight loading
</syntaxhighlight>

=== BitsAndBytes Preprocessing (Example Implementation) ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer
import torch

class Bnb4BitQuantizer(HfQuantizer):
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.pre_quantized = False  # BnB does on-the-fly quantization

    def _process_model_before_weight_loading(self, model, **kwargs):
        """BitsAndBytes-specific preprocessing"""
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device

        # Set module-level flags for quantization
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Mark for later conversion to Linear4bit
                module._is_hf_initialized = False

        # Adjust dtype if needed
        if self.quantization_config.bnb_4bit_compute_dtype:
            model._compute_dtype = self.quantization_config.bnb_4bit_compute_dtype

        return model

# Usage
quantizer = Bnb4BitQuantizer(BitsAndBytesConfig(load_in_4bit=True))
quantizer.preprocess_model(model)
# Model's Linear layers are flagged for 4-bit conversion during loading
</syntaxhighlight>

=== GPTQ Preprocessing with Module Conversion ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer
import torch

class GptqQuantizer(HfQuantizer):
    requires_calibration = True

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.pre_quantized = True  # GPTQ loads pre-quantized weights

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Replace Linear modules with QuantLinear"""
        from auto_gptq.nn_modules.qlinear import QuantLinear
        from accelerate.utils import get_module_from_name

        modules_to_not_convert = self.get_modules_to_not_convert(model)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name not in modules_to_not_convert:
                # Replace with GPTQ quantized linear layer
                parent, child_name = get_module_from_name(model, name)

                quant_linear = QuantLinear(
                    bits=self.quantization_config.bits,
                    group_size=self.quantization_config.group_size,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                )

                # Replace in parent module
                setattr(parent, child_name, quant_linear)

        return model

# Usage
from transformers import GPTQConfig

gptq_config = GPTQConfig(bits=4, group_size=128)
quantizer = GptqQuantizer(gptq_config, pre_quantized=True)
quantizer.preprocess_model(model)

# Model's Linear layers are now QuantLinear instances
# Ready to load pre-quantized INT4 weights
</syntaxhighlight>

=== Custom Quantizer with Skip Modules ===
<syntaxhighlight lang="python">
from transformers.quantizers.base import HfQuantizer
import torch

class CustomQuantizer(HfQuantizer):
    def _process_model_before_weight_loading(self, model, **kwargs):
        """Quantize all Linear layers except those in skip list"""
        skip_modules = self.quantization_config.skip_modules or []
        modules_to_not_convert = self.get_modules_to_not_convert(
            model,
            skip_modules=skip_modules,
            add_default_skips=True
        )

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in modules_to_not_convert:
                    # Keep in full precision
                    module._keep_in_fp32 = True
                    print(f"Skipping quantization: {name}")
                else:
                    # Mark for quantization
                    module._quantize = True
                    module._quant_config = self.quantization_config
                    print(f"Will quantize: {name}")

        return model

# Usage
from transformers import QuantizationConfig

config = QuantizationConfig(
    quant_method="custom",
    skip_modules=["lm_head", "model.embed_tokens"]
)
quantizer = CustomQuantizer(config)
quantizer.preprocess_model(model)

# lm_head and embed_tokens remain full precision
# All other Linear layers are marked for quantization
</syntaxhighlight>

=== Preprocessing with Device Map Adjustment ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import infer_auto_device_map, init_empty_weights
import torch

# Step 1: Initialize on meta device
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-hf",
        torch_dtype=torch.float16,
    )

# Step 2: Create quantizer and preprocess
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
quantizer = AutoHfQuantizer.from_config(quant_config)
quantizer.preprocess_model(model, dtype=torch.float16)

# Step 3: Compute device map with quantization memory estimates
# After preprocessing, memory estimates are adjusted for 4-bit
device_map = infer_auto_device_map(
    model,
    max_memory={0: "20GB"},  # 13B model fits in 20GB with 4-bit
    dtype=torch.float16,
    no_split_module_classes=["LlamaDecoderLayer"],
)

# Model ready for weight loading with correct device placement
</syntaxhighlight>

=== Complete Preprocessing Pipeline ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.quantizers.auto import AutoHfQuantizer
from accelerate import init_empty_weights
import torch

def load_quantized_model(model_name, quantization_config):
    """Complete preprocessing pipeline"""

    # Step 1: Load config
    config = AutoConfig.from_pretrained(model_name)

    # Step 2: Initialize empty model on meta device
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    print(f"Model initialized on {model.device}")  # meta

    # Step 3: Create quantizer
    quantizer = AutoHfQuantizer.from_config(quantization_config)

    # Step 4: Validate environment
    quantizer.validate_environment(device_map="auto")

    # Step 5: Preprocess model
    quantizer.preprocess_model(model, dtype=torch.float16)

    print(f"Model quantization flags set:")
    print(f"  is_quantized: {model.is_quantized}")
    print(f"  quantization_method: {model.quantization_method}")

    # Step 6: Load weights (not shown - uses preprocessed model)
    # load_checkpoint_and_dispatch(model, checkpoint, device_map=...)

    return model

# Usage
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = load_quantized_model("meta-llama/Llama-2-7b-hf", config)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Weight_Quantization]]

=== Requires ===
* [[requires::Implementation:huggingface_transformers_Quantizer_validate_environment]]

=== Precedes ===
* [[precedes::Implementation:huggingface_transformers_Quantizer_convert_weights]]
* [[precedes::Implementation:huggingface_transformers_Quantizer_postprocess]]
