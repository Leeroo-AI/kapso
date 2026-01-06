{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Quantization]], [[domain::Model Compression]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
IA3 quantized layer implementations (Linear8bitLt and Linear4bit) provide parameter-efficient fine-tuning with bitsandbytes quantization support, enabling (IA)³ adaptation on quantized models for reduced memory footprint.

=== Description ===
This module provides two quantized implementations of (IA)³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) layers that work with bitsandbytes quantization:

* '''Linear8bitLt''' - (IA)³ for 8-bit quantized linear layers
* '''Linear4bit''' - (IA)³ for 4-bit quantized linear layers

Both implementations extend the IA3Layer base class and wrap quantized base layers from bitsandbytes. They apply learned scaling vectors to either inputs (for feedforward layers) or outputs (for attention layers), enabling efficient adaptation of quantized models with minimal memory overhead. The quantized base layer weights remain frozen during training.

=== Usage ===
Use these quantized IA3 layers when you need:
* Fine-tuning large models with limited GPU memory
* (IA)³ adaptation combined with quantization for maximum efficiency
* Training on consumer hardware with memory constraints
* Inference optimization while maintaining adaptation capability
* Reduced memory footprint during both training and inference

== Code Reference ==
=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/bnb.py src/peft/tuners/ia3/bnb.py]
* '''Lines:''' 26-130

=== Signature ===
<syntaxhighlight lang="python">
class Linear8bitLt(torch.nn.Module, IA3Layer):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        is_feedforward: bool,
        init_ia3_weights: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            base_layer: 8-bit quantized linear layer to wrap
            adapter_name: Name of the adapter
            is_feedforward: Whether this is a feedforward layer
            init_ia3_weights: Whether to initialize IA3 weights
        """

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply IA3 scaling to quantized layer"""


class Linear4bit(torch.nn.Module, IA3Layer):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        is_feedforward: bool,
        init_ia3_weights: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            base_layer: 4-bit quantized linear layer to wrap
            adapter_name: Name of the adapter
            is_feedforward: Whether this is a feedforward layer
            init_ia3_weights: Whether to initialize IA3 weights
        """

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply IA3 scaling to quantized layer"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# These classes are used internally by PEFT
# Users typically access them through the high-level API
from peft import IA3Config, get_peft_model, prepare_model_for_kbit_training
</syntaxhighlight>

== I/O Contract ==
=== Input Parameters (Constructor) ===
{| class="wikitable"
! Parameter !! Type !! Description !! Default
|-
| base_layer || torch.nn.Module || The quantized linear layer to wrap || Required
|-
| adapter_name || str || Name identifier for the adapter || Required
|-
| is_feedforward || bool || Whether layer is feedforward (scales input) or attention (scales output) || Required
|-
| init_ia3_weights || bool || Whether to initialize IA3 scaling vectors || True
|-
| **kwargs || dict || Additional keyword arguments || -
|}

=== Forward Method ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| x || torch.Tensor || Input tensor to the layer
|-
| *args || tuple || Additional positional arguments passed to base layer
|-
| **kwargs || dict || Additional keyword arguments passed to base layer
|}

=== Output ===
{| class="wikitable"
! Return Type !! Description
|-
| torch.Tensor || Output tensor with IA3 scaling applied
|}

=== Attributes (Inherited from IA3Layer) ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| ia3_l || torch.nn.ParameterDict || Dictionary of learned scaling vectors per adapter
|-
| is_feedforward || bool || Whether to scale inputs (True) or outputs (False)
|-
| active_adapters || list || List of currently active adapter names
|-
| disable_adapters || bool || Flag to disable all adapters
|}

== Usage Examples ==
=== Basic Quantized IA3 Model (8-bit) ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import IA3Config, get_peft_model, prepare_model_for_kbit_training
import torch

# Load model with 8-bit quantization
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Configure IA3
config = IA3Config(
    target_modules=["q_proj", "v_proj"],
    feedforward_modules=["fc2"],  # Apply to inputs for feedforward
    init_ia3_weights=True
)

# Apply IA3 (automatically uses Linear8bitLt for quantized layers)
model = get_peft_model(model, config)

# Check trainable parameters
model.print_trainable_parameters()
# Output shows very few trainable parameters despite large model
</syntaxhighlight>

=== 4-bit Quantized IA3 Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import IA3Config, get_peft_model, prepare_model_for_kbit_training

# Load model with 4-bit quantization (even more memory efficient)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare and apply IA3
model = prepare_model_for_kbit_training(base_model)

config = IA3Config(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    feedforward_modules=["gate_proj", "up_proj", "down_proj"],
    init_ia3_weights=True
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== Training with Quantized IA3 ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import IA3Config, get_peft_model, prepare_model_for_kbit_training

# Setup
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# Apply IA3
config = IA3Config(
    target_modules=["q_proj", "v_proj"],
    feedforward_modules=["fc2"],
    init_ia3_weights=True
)
model = get_peft_model(model, config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./ia3_quantized",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save only the IA3 adapters (base model stays quantized)
model.save_pretrained("./ia3_quantized_adapter")
</syntaxhighlight>

=== Inference with Quantized IA3 ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load quantized base model
base_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto"
)

# Load IA3 adapter
model = PeftModel.from_pretrained(base_model, "./ia3_quantized_adapter")

# Inference
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
</syntaxhighlight>

=== Multiple Quantized Adapters ===
<syntaxhighlight lang="python">
# Load base quantized model
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# Add first adapter
config1 = IA3Config(target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config1, adapter_name="task1")

# Add second adapter
config2 = IA3Config(target_modules=["q_proj", "v_proj"])
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
# ... inference for task1 ...

model.set_adapter("task2")
# ... inference for task2 ...
</syntaxhighlight>

=== Memory-Efficient Large Model Fine-tuning ===
<syntaxhighlight lang="python">
# Fine-tune 13B model on a single GPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    load_in_4bit=True,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
)

model = prepare_model_for_kbit_training(model)

# Minimal IA3 configuration
config = IA3Config(
    target_modules=["q_proj", "v_proj"],
    feedforward_modules=["down_proj"]
)

model = get_peft_model(model, config)

# Can now train 13B model on 24GB GPU
</syntaxhighlight>

== Technical Details ==
=== Quantization Support ===
* '''8-bit (Linear8bitLt)''': Uses bitsandbytes LLM.int8() quantization
* '''4-bit (Linear4bit)''': Uses bitsandbytes 4-bit NormalFloat quantization

=== Memory Efficiency ===
* Base weights remain frozen and quantized
* Only IA3 scaling vectors are trained (FP32/FP16)
* Typical memory reduction: 75% (4-bit) or 50% (8-bit) vs FP16

=== Forward Pass Behavior ===
For feedforward layers (is_feedforward=True):
1. Scale input: x_scaled = x * ia3_scaling
2. Apply quantized layer: output = base_layer(x_scaled)

For attention layers (is_feedforward=False):
1. Apply quantized layer: output = base_layer(x)
2. Scale output: output_scaled = output * ia3_scaling

=== Type Conversion ===
Both implementations handle automatic type conversion:
* Convert to FP32 when not in autocast mode
* Convert back to expected dtype after computation
* Special handling for 4-bit to support older PyTorch versions

=== Dependencies ===
* Requires bitsandbytes library
* Check availability with is_bnb_available() and is_bnb_4bit_available()

== Related Pages ==
* [[uses::Component:huggingface_peft_IA3Layer]]
* [[configured_by::Configuration:huggingface_peft_IA3Config]]
* [[related::Component:huggingface_peft_IA3Model]]
* [[requires::Library:bitsandbytes]]
* [[related::Component:huggingface_peft_LoRA_Quantized]]
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
* [[implements::Technique:IA3_with_Quantization]]
