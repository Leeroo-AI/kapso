= VeraModel =

== Knowledge Sources ==

* '''Repository''': [https://github.com/huggingface/peft HuggingFace PEFT]
* '''Paper''': [https://huggingface.co/papers/2310.11454 VeRA: Vector-based Random Matrix Adaptation]
* '''Type''': Model Class
* '''Module''': peft.tuners.vera.model

== Domains ==

[[Category:Natural_Language_Processing]]
[[Category:Parameter_Efficient_Fine_Tuning]]
[[Category:Vector_Adaptation]]
[[Category:Low_Rank_Adaptation]]
[[Category:Model_Architecture]]

== Overview ==

=== Description ===

VeraModel creates Vector-based Random Matrix Adaptation (VeRA) models from pretrained transformers. VeRA is a parameter-efficient fine-tuning technique that achieves significant parameter reduction compared to LoRA by using shared random projection matrices (vera_A and vera_B) across all adapted layers.

The key innovation is that instead of having separate low-rank matrices per layer like LoRA, VeRA uses:
* Shared frozen random matrices vera_A and vera_B (initialized with Kaiming uniform using a PRNG seed)
* Small trainable scaling vectors lambda_b and lambda_d per layer

This allows VeRA to use higher rank values (e.g., 256) while maintaining fewer trainable parameters than LoRA with lower ranks.

=== Usage ===

VeraModel extends BaseTuner and handles the injection of VeRA layers into pretrained models. It manages the initialization and sharing of projection matrices, creates appropriate layer types (including quantized variants), and ensures consistent configuration across adapters.

== Code Reference ==

=== Source Location ===

<code>/tmp/praxium_repo_zyf9ywdz/src/peft/tuners/vera/model.py</code>

=== Signature ===

<syntaxhighlight lang="python">
class VeraModel(BaseTuner):
    def __init__(
        self,
        model: PreTrainedModel,
        config: VeraConfig,
        adapter_name: str = "default",
        low_cpu_mem_usage: bool = False
    )
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from peft import VeraModel, VeraConfig
# Or use the high-level API
from peft import get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| model || PreTrainedModel || Required || The pretrained transformers model to adapt
|-
| config || VeraConfig || Required || VeRA configuration object
|-
| adapter_name || str || "default" || Name for the adapter
|-
| low_cpu_mem_usage || bool || False || Create empty adapter weights on meta device for faster loading
|}

=== Key Attributes ===

{| class="wikitable"
! Attribute !! Type !! Description
|-
| prefix || str || "vera_lambda_" - prefix for VeRA parameters
|-
| tuner_layer_cls || class || VeraLayer - the layer class used for VeRA
|-
| vera_A || BufferDict || Shared projection matrix A (r × in_features)
|-
| vera_B || BufferDict || Shared projection matrix B (out_features × r)
|-
| target_module_mapping || dict || Mapping of model types to default target modules
|}

=== Return Values ===

{| class="wikitable"
! Method !! Return Type !! Description
|-
| __init__ || VeraModel || Initialized VeRA model ready for training
|-
| _find_dim || tuple[int, int] || Largest (out_features, in_features) across target layers
|-
| _init_vera_A_vera_B || None || Initializes shared projection matrices
|}

== Usage Examples ==

=== Basic VeRA Model Creation ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import VeraConfig, get_peft_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Create VeRA configuration
config = VeraConfig(
    r=128,
    target_modules=["q_proj", "v_proj"],
    vera_dropout=0.1
)

# Create VeRA model
model = get_peft_model(base_model, config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: ~200K || all params: 125M || trainable%: 0.16%
</syntaxhighlight>

=== Using with Quantized Models ===

<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import VeraConfig, get_peft_model
import torch

# Load 4-bit quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply VeRA to quantized model
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    vera_dropout=0.05,
    d_initial=0.1
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Multiple Adapter Configuration ===

<syntaxhighlight lang="python">
from peft import VeraConfig, get_peft_model

# Create model with first adapter
config1 = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    projection_prng_key=42  # Must be same for all adapters
)
model = get_peft_model(base_model, config1, adapter_name="task1")

# Add second adapter (shares vera_A and vera_B)
config2 = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    projection_prng_key=42,  # Same key required
    vera_dropout=0.2
)
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
# ... train or infer
model.set_adapter("task2")
# ... train or infer
</syntaxhighlight>

=== Layer-Specific Adaptation ===

<syntaxhighlight lang="python">
from peft import VeraConfig, get_peft_model

# Apply VeRA only to specific transformer layers
config = VeraConfig(
    r=512,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_to_transform=[0, 1, 2],  # Only first 3 layers
    layers_pattern="layers",  # Depends on model architecture
    vera_dropout=0.1,
    d_initial=0.05,
    save_projection=True
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Training and Saving ===

<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
from peft import VeraConfig, get_peft_model

# Create and train VeRA model
config = VeraConfig(r=256, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)

training_args = TrainingArguments(
    output_dir="./vera_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Save VeRA adapter weights
model.save_pretrained("./vera_adapter")

# Load later
from peft import PeftModel
loaded_model = PeftModel.from_pretrained(base_model, "./vera_adapter")
</syntaxhighlight>

=== Custom Initialization ===

<syntaxhighlight lang="python">
from peft import VeraConfig, get_peft_model

# Use specific PRNG key for reproducible initialization
config = VeraConfig(
    r=256,
    target_modules=["q_proj", "v_proj"],
    projection_prng_key=12345,  # Deterministic initialization
    d_initial=0.1,  # Small initial scaling
    save_projection=True  # Save vera_A and vera_B in checkpoint
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

== Implementation Details ==

=== Kaiming Initialization ===

VeRA uses a custom Kaiming uniform initialization for the shared projection matrices:

<syntaxhighlight lang="python">
def _kaiming_init(tensor_or_shape, generator):
    """
    Kaiming Uniform with PRNG generator for deterministic initialization.
    fan = in_features, gain = sqrt(2), bound = sqrt(3) * gain / sqrt(fan)
    """
    # Returns tensor uniformly distributed in [-bound, bound]
</syntaxhighlight>

=== Dimension Finding ===

The model automatically finds the largest dimensions across all target layers to size the shared projection matrices:

<syntaxhighlight lang="python">
largest_out_dim, largest_in_dim = model._find_dim(config)
vera_A = [r × largest_in_dim]
vera_B = [largest_out_dim × r]
</syntaxhighlight>

=== Supported Layer Types ===

* <code>torch.nn.Linear</code> - Standard linear layers
* <code>Conv1D</code> - GPT-2 style convolution layers
* <code>bnb.nn.Linear8bitLt</code> - 8-bit quantized layers
* <code>bnb.nn.Linear4bit</code> - 4-bit quantized layers

== Related Pages ==

* [[huggingface_peft_VeraConfig|VeraConfig]] - Configuration class for VeRA
* [[huggingface_peft_VeraLayer|VeraLayer]] - Layer implementation
* [[huggingface_peft_VeraQuantized|VeraQuantized]] - Quantized VeRA variants
* [[huggingface_peft_LoraModel|LoraModel]] - Similar LoRA model class
* [[huggingface_peft_BaseTuner|BaseTuner]] - Base class for PEFT tuners
* [[Parameter_Efficient_Fine_Tuning|PEFT Overview]]
