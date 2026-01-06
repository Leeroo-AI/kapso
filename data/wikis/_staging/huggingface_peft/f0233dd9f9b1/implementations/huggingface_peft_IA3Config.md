{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
IA3Config is the configuration class for (IA)³ (Infused Adapter by Inhibiting and Amplifying Inner Activations), a parameter-efficient fine-tuning method that learns element-wise scaling vectors for model adaptation.

=== Description ===
IA3Config stores the configuration parameters for (IA)³ models, which adapt pretrained models by learning scaling vectors that are multiplied element-wise with layer activations. Unlike LoRA which adds trainable rank decomposition matrices, (IA)³ multiplies learned vectors to either inputs (for feedforward layers) or outputs (for attention layers), resulting in even fewer trainable parameters.

The configuration is a dataclass that extends PeftConfig and includes validation logic to ensure feedforward_modules is a subset of target_modules. It distinguishes between attention and feedforward layers to apply scaling at the appropriate point in the computation.

=== Usage ===
Use IA3Config when you need:
* Extremely parameter-efficient fine-tuning (fewer parameters than LoRA)
* Element-wise scaling adaptation for model customization
* Different treatment for attention vs feedforward layers
* Fast adaptation with minimal memory overhead
* Compatible alternative to full fine-tuning or LoRA

== Code Reference ==
=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/config.py src/peft/tuners/ia3/config.py]
* '''Lines:''' 24-113

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class IA3Config(PeftConfig):
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    feedforward_modules: Optional[Union[list[str], str]] = None
    fan_in_fan_out: bool = False
    modules_to_save: Optional[list[str]] = None
    init_ia3_weights: bool = True
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import IA3Config
</syntaxhighlight>

== I/O Contract ==
=== Configuration Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| target_modules || Union[list[str], str] || None || Module names or regex to apply (IA)³ to (e.g., ['q', 'v'] or 'all-linear')
|-
| exclude_modules || Union[list[str], str] || None || Module names or regex to exclude from (IA)³
|-
| feedforward_modules || Union[list[str], str] || None || Modules to treat as feedforward (scale inputs instead of outputs)
|-
| fan_in_fan_out || bool || False || True if layer stores weights as (fan_in, fan_out) like Conv1D
|-
| modules_to_save || list[str] || None || Additional modules to be trainable and saved
|-
| init_ia3_weights || bool || True || Whether to initialize (IA)³ scaling vectors
|}

=== Validation Rules ===
{| class="wikitable"
! Rule !! Description
|-
| Feedforward subset || feedforward_modules must be a subset of target_modules
|-
| Set conversion || Converts list target_modules, exclude_modules, and feedforward_modules to sets
|-
| Initialization recommendation || init_ia3_weights=False is discouraged
|}

=== Output ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| peft_type || PeftType || Set to PeftType.IA3 after initialization
|}

== Usage Examples ==
=== Basic IA3 Configuration ===
<syntaxhighlight lang="python">
from peft import IA3Config, get_peft_model
from transformers import AutoModelForCausalLM

# Basic configuration
config = IA3Config(
    target_modules=["q_proj", "v_proj"],
    feedforward_modules=[],  # Treat as attention layers (scale outputs)
    init_ia3_weights=True
)

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = get_peft_model(base_model, config)

# Very few trainable parameters
model.print_trainable_parameters()
</syntaxhighlight>

=== With Feedforward Modules ===
<syntaxhighlight lang="python">
# Distinguish between attention and feedforward layers
config = IA3Config(
    target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    feedforward_modules=["fc1", "fc2"],  # Scale inputs for these
    init_ia3_weights=True
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Wildcard Target All Linear ===
<syntaxhighlight lang="python">
# Target all linear layers except output
config = IA3Config(
    target_modules="all-linear",
    feedforward_modules=["mlp.fc1", "mlp.fc2"],
    init_ia3_weights=True
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Regex Pattern Targeting ===
<syntaxhighlight lang="python">
# Use regex to target specific layers
config = IA3Config(
    target_modules=r".*decoder.*(SelfAttention|EncDecAttention).*(q|v)$",
    feedforward_modules=r".*decoder.*fc2$",
    init_ia3_weights=True
)
</syntaxhighlight>

=== Excluding Specific Modules ===
<syntaxhighlight lang="python">
# Target many layers but exclude some
config = IA3Config(
    target_modules="all-linear",
    exclude_modules=["lm_head", "embed_tokens"],
    feedforward_modules=["fc1", "fc2"],
    init_ia3_weights=True
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Conv1D Layer Configuration (GPT-2) ===
<syntaxhighlight lang="python">
# For models using Conv1D like GPT-2
config = IA3Config(
    target_modules=["c_attn", "c_proj", "c_fc"],
    feedforward_modules=["c_fc"],
    fan_in_fan_out=True,  # Required for Conv1D
    init_ia3_weights=True
)

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Sequence Classification ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSequenceClassification

# Fine-tune for classification
config = IA3Config(
    target_modules=["q_proj", "v_proj"],
    feedforward_modules=["fc2"],
    modules_to_save=["classifier"],  # Also train classifier head
    init_ia3_weights=True
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Minimal Configuration ===
<syntaxhighlight lang="python">
# Absolute minimal parameter count
config = IA3Config(
    target_modules=["q_proj", "v_proj"],  # Only attention query and value
    init_ia3_weights=True
)

model = get_peft_model(base_model, config)
# Extremely few trainable parameters
</syntaxhighlight>

=== Token Classification ===
<syntaxhighlight lang="python">
from transformers import AutoModelForTokenClassification

config = IA3Config(
    target_modules=["q_proj", "v_proj"],
    feedforward_modules=["fc2"],
    modules_to_save=["classifier"],
    init_ia3_weights=True
)

base_model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=9  # e.g., NER tags
)
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Large Language Model Configuration ===
<syntaxhighlight lang="python">
# For Llama-style models
config = IA3Config(
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    feedforward_modules=["gate_proj", "up_proj", "down_proj"],
    init_ia3_weights=True
)

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Quantized Model Configuration ===
<syntaxhighlight lang="python">
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Load quantized model
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(base_model)

# IA3 works great with quantization
config = IA3Config(
    target_modules=["q_proj", "v_proj"],
    feedforward_modules=["fc2"],
    init_ia3_weights=True
)

model = get_peft_model(model, config)
</syntaxhighlight>

== Technical Details ==
=== (IA)³ Scaling Mechanism ===
* '''Attention layers''' (not in feedforward_modules):
  - Scaling applied to outputs
  - output = base_layer(input) * ia3_vector

* '''Feedforward layers''' (in feedforward_modules):
  - Scaling applied to inputs
  - output = base_layer(input * ia3_vector)

=== Parameter Count ===
For a layer with output dimension d:
* (IA)³ parameters: d (a single scaling vector)
* Much fewer than LoRA: d << 2 * d * r

=== Initialization ===
When init_ia3_weights=True:
* Scaling vectors initialized to ones
* Model initially behaves like the base model
* Discouraging to set to False

=== Target Module Specification ===
* '''List''': Exact match or suffix match
* '''String''': Regex pattern matching
* '''"all-linear"''': All Linear/Conv1D except output layer

== Related Pages ==
* [[configures::Component:huggingface_peft_IA3Model]]
* [[inherits_from::Configuration:huggingface_peft_PeftConfig]]
* [[uses::Enumeration:huggingface_peft_PeftType]]
* [[related::Configuration:huggingface_peft_LoraConfig]]
* [[related::Configuration:huggingface_peft_AdaLoraConfig]]
* [[related::Component:huggingface_peft_IA3Quantized]]
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
* [[implements::Technique:IA3_Adaptation]]
