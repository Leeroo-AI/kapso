= VBLoRAConfig =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* [https://huggingface.co/papers/2405.15179 VB-LoRA Paper]
* Source: src/peft/tuners/vblora/config.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Low-Rank Adaptation]]
* [[Vector Quantization]]
* [[Model Compression]]

== Overview ==

=== Description ===
VBLoRAConfig is the configuration class for Vector Bank Low-Rank Adaptation (VB-LoRA), an advanced parameter-efficient fine-tuning method. VB-LoRA uses a shared vector bank to construct low-rank adaptation matrices, significantly reducing the number of parameters that need to be saved compared to standard LoRA while maintaining similar or better performance.

The method uses learnable logits to select vectors from a shared bank, enabling efficient parameter sharing across different layers and adapters. The configuration controls the vector bank size, rank, top-K selection, and various initialization and training parameters.

=== Usage ===
VBLoRAConfig is used to configure VB-LoRA adapters for pre-trained models. It extends PeftConfig and provides VB-LoRA-specific parameters for controlling the vector bank, selection mechanism, and adaptation behavior. The configuration is passed to get_peft_model() to create a VB-LoRA adapted model.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/vblora/config.py
Lines: 25-197

=== Class Signature ===
<syntaxhighlight lang="python">
@dataclass
class VBLoRAConfig(PeftConfig):
    """
    Configuration class for VBLoRAModel.

    Key Parameters:
        r: Rank of incremental matrices (default: 4)
        num_vectors: Number of vectors in vector bank (default: 256)
        vector_length: Length of each vector (default: 256)
        topk: K value for top-K selection (default: 2)
        save_only_topk_weights: Save only topk weights for inference (default: False)
    """
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.vblora.config import VBLoRAConfig
# Or via the main PEFT interface
from peft import VBLoRAConfig
</syntaxhighlight>

== I/O Contract ==

=== Configuration Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| r || int || 4 || Rank of incremental matrices
|-
| num_vectors || int || 256 || Number of vectors in the vector bank
|-
| vector_length || int || 256 || Length of vectors; must divide hidden dimension
|-
| topk || int || 2 || K value for top-K selection
|-
| target_modules || Optional[Union[list[str], str]] || None || Modules to replace with VB-LoRA
|-
| exclude_modules || Optional[Union[list[str], str]] || None || Modules to exclude from VB-LoRA
|-
| save_only_topk_weights || bool || False || Save only topk weights (inference only)
|-
| vblora_dropout || float || 0.0 || Dropout probability for VB-LoRA layers
|-
| fan_in_fan_out || bool || False || True if layer stores weight as (fan_in, fan_out)
|-
| bias || str || "none" || Bias type: 'none', 'all', or 'vblora_only'
|-
| modules_to_save || Optional[list[str]] || None || Additional trainable modules
|-
| init_vector_bank_bound || float || 0.02 || Uniform distribution bound for vector bank init
|-
| init_logits_std || float || 0.1 || Standard deviation for logits initialization
|-
| layers_to_transform || Optional[Union[list[int], int]] || None || Specific layer indices to transform
|-
| layers_pattern || Optional[Union[list[str], str]] || None || Layer pattern name for layer selection
|}

=== Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| peft_type || PeftType || Set to PeftType.VBLORA in __post_init__
|}

== Usage Examples ==

=== Basic VB-LoRA Configuration ===
<syntaxhighlight lang="python">
from peft import VBLoRAConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create VB-LoRA configuration
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "v_proj"]
)

# Apply VB-LoRA
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
</syntaxhighlight>

=== High-Performance Configuration ===
<syntaxhighlight lang="python">
# Configuration optimized for performance
config = VBLoRAConfig(
    r=8,
    num_vectors=512,  # Larger bank for bigger models
    vector_length=128,
    topk=2,  # K=2 provides best balance
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    vblora_dropout=0.1
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Inference-Optimized Configuration ===
<syntaxhighlight lang="python">
# Save only top-k weights for minimal storage
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    topk=2,
    save_only_topk_weights=True,  # Significantly reduces storage
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, config)
# Note: Models with save_only_topk_weights=True cannot resume training
</syntaxhighlight>

=== Target All Linear Layers ===
<syntaxhighlight lang="python">
# Apply to all linear layers except output
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    target_modules="all-linear",
    exclude_modules=["lm_head"]
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Specific Layer Selection ===
<syntaxhighlight lang="python">
# Apply VB-LoRA to specific layers only
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3, 4],  # First 5 layers only
    layers_pattern="layers"
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Custom Initialization ===
<syntaxhighlight lang="python">
# Custom initialization parameters
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    init_vector_bank_bound=0.01,  # Smaller init for stability
    init_logits_std=0.05,  # Smaller std for logits
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== With Bias Training ===
<syntaxhighlight lang="python">
# Train biases along with VB-LoRA
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    target_modules=["q_proj", "v_proj"],
    bias="all",  # Train all biases
    modules_to_save=["classifier"]  # Also save classifier
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Regex Target Selection ===
<syntaxhighlight lang="python">
# Use regex to select target modules
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    target_modules=r".*decoder.*(self_attn|encoder_attn).*(q|v)_proj$"
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Full Training Configuration ===
<syntaxhighlight lang="python">
from transformers import TrainingArguments, Trainer

# VB-LoRA config
config = VBLoRAConfig(
    r=4,
    num_vectors=256,
    vector_length=256,
    topk=2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    vblora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vblora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_VBLoRALayer|VBLoRALayer]] - Layer implementation for VB-LoRA
* [[huggingface_peft_VBLoRAModel|VBLoRAModel]] - Model class for VB-LoRA
* [[huggingface_peft_LoraConfig|LoraConfig]] - Configuration for standard LoRA
* [[Low-Rank Adaptation]]
* [[Vector Quantization]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Model Configuration]]
[[Category:HuggingFace]]
