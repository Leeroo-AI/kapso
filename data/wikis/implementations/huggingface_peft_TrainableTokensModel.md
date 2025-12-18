= TrainableTokensModel =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* Source: src/peft/tuners/trainable_tokens/model.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Transfer Learning]]
* [[Token Embeddings]]
* [[Domain Adaptation]]

== Overview ==

=== Description ===
TrainableTokensModel is the main model class that manages the injection and coordination of TrainableTokensLayer adapters into pre-trained models. It extends BaseTuner and handles the automatic detection of embedding layers, creation of trainable token adapters, and management of weight-tied scenarios where input and output embeddings share parameters.

The model automatically infers the embedding layer name if not specified and supports weight tying between embedding and LM head layers, ensuring consistent updates across tied weights.

=== Usage ===
TrainableTokensModel is instantiated through get_peft_model() by passing a TrainableTokensConfig. It automatically identifies embedding layers (or uses specified target_modules), wraps them with TrainableTokensLayer, and handles weight tying for models where input embeddings and output layers share parameters.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/trainable_tokens/model.py
Lines: 26-140

=== Class Signature ===
<syntaxhighlight lang="python">
class TrainableTokensModel(BaseTuner):
    """
    Model class for TrainableTokens method.

    Automatically handles:
    - Embedding layer detection
    - Weight tying between input/output embeddings
    - Adapter injection and management
    """
    prefix: str = "trainable_tokens_"
    tuner_layer_cls = TrainableTokensLayer
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.trainable_tokens.model import TrainableTokensModel
# Or via the main PEFT interface
from peft import get_peft_model, TrainableTokensConfig
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| model || nn.Module || The base model to adapt
|-
| config || TrainableTokensConfig || Configuration for trainable tokens
|-
| adapter_name || str || Name of the adapter (default: "default")
|}

=== Class Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| prefix || str || Prefix for trainable tokens parameters ("trainable_tokens_")
|-
| tuner_layer_cls || Type || TrainableTokensLayer class reference
|}

=== Key Methods ===
{| class="wikitable"
! Method !! Parameters !! Returns !! Description
|-
| _prepare_adapter_config || peft_config, model_config || PeftConfig || Prepare config, infer embedding layer if needed
|-
| inject_adapter || model, adapter_name, autocast_adapter_dtype, low_cpu_mem_usage, kwargs || None || Inject adapter and handle weight tying
|-
| _get_tied_target_modules || *args, **kwargs || list || Override to suppress tied weights warning
|-
| _create_and_replace || peft_config, adapter_name, target, target_name, parent, current_key || None || Create and replace module with adapter
|-
| _create_and_replace_dict || peft_config, adapter_name, target, target_name, parent, current_key || None || Create and replace with dict config
|-
| _create_new_module || peft_config, adapter_name, target, kwargs || TrainableTokensLayer || Create new TrainableTokensLayer
|}

== Usage Examples ==

=== Basic Model Creation ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TrainableTokensConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Add new tokens
new_tokens = ["<special1>", "<special2>"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Get token indices
token_indices = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]

# Create trainable tokens model
config = TrainableTokensConfig(token_indices=token_indices)
model = get_peft_model(model, config)

print(model.print_trainable_parameters())
</syntaxhighlight>

=== Auto-detection of Embedding Layer ===
<syntaxhighlight lang="python">
# Without specifying target_modules, it auto-detects embedding layer
config = TrainableTokensConfig(
    token_indices=[100, 101, 102]
    # target_modules not specified - will use get_input_embeddings()
)

model = get_peft_model(base_model, config)

# Check which layer was targeted
print(f"Input embeddings: {model.get_input_embeddings()}")
</syntaxhighlight>

=== Explicit Target Modules ===
<syntaxhighlight lang="python">
# Specify embedding layer explicitly
config = TrainableTokensConfig(
    token_indices=[100, 101, 102],
    target_modules=["model.embed_tokens"]
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Weight Tying Support ===
<syntaxhighlight lang="python">
# For models with tied input/output embeddings
# (e.g., where lm_head shares weights with embeddings)

config = TrainableTokensConfig(
    token_indices=[50000, 50001, 50002]
)

# Automatically handles tied weights
model = get_peft_model(base_model, config)

# Both input embeddings and lm_head are adapted with tied weights
print("Input embeddings:", type(model.get_input_embeddings()))
print("Output embeddings:", type(model.get_output_embeddings()))
</syntaxhighlight>

=== Training New Domain Vocabulary ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Add domain-specific tokens
domain_tokens = ["<medical>", "<diagnosis>", "<treatment>"]
tokenizer.add_tokens(domain_tokens)
model.resize_token_embeddings(len(tokenizer))

# Configure trainable tokens
token_indices = [tokenizer.convert_tokens_to_ids(t) for t in domain_tokens]
config = TrainableTokensConfig(
    token_indices=token_indices,
    init_weights=False  # Random init for new tokens
)

model = get_peft_model(model, config)

# Training
training_args = TrainingArguments(
    output_dir="./domain_adapted_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
</syntaxhighlight>

=== Multiple Adapters for Different Tasks ===
<syntaxhighlight lang="python">
# First adapter for task 1
config1 = TrainableTokensConfig(token_indices=[100, 101, 102])
model = get_peft_model(base_model, config1, adapter_name="task1")

# Second adapter for task 2 (different tokens)
config2 = TrainableTokensConfig(token_indices=[200, 201, 202])
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
output1 = model(input_ids)

model.set_adapter("task2")
output2 = model(input_ids)
</syntaxhighlight>

=== Save and Load ===
<syntaxhighlight lang="python">
# Save adapter
model.save_pretrained("./trainable_tokens_adapter")

# Load adapter
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = PeftModel.from_pretrained(base_model, "./trainable_tokens_adapter")
</syntaxhighlight>

=== Merge and Unload ===
<syntaxhighlight lang="python">
# Merge trainable tokens into base model
model = model.merge_and_unload()

# Now it's a standard model with updated embeddings
model.save_pretrained("./merged_model")
</syntaxhighlight>

=== Checking Trainable Parameters ===
<syntaxhighlight lang="python">
# Print parameter statistics
model.print_trainable_parameters()

# Manually inspect
trainable_params = 0
all_params = 0
for name, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"Trainable: {name}, Shape: {param.shape}")

print(f"Trainable: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_TrainableTokensConfig|TrainableTokensConfig]] - Configuration class
* [[huggingface_peft_TrainableTokensLayer|TrainableTokensLayer]] - Layer implementation
* [[Token Embeddings]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Domain Adaptation]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Model Adaptation]]
[[Category:NLP]]
[[Category:HuggingFace]]
