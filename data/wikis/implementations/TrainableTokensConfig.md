= TrainableTokensConfig =

== Knowledge Sources ==
* [https://github.com/huggingface/peft HuggingFace PEFT Repository]
* Source: src/peft/tuners/trainable_tokens/config.py

== Domains ==
* [[Natural Language Processing (NLP)]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Token Embeddings]]
* [[Transfer Learning]]
* [[Memory Optimization]]

== Overview ==

=== Description ===
TrainableTokensConfig is a configuration class for the TrainableTokens method, which enables training new tokens or re-training existing tokens without training the full embedding matrix. This method marks select tokens (identified by their indices) as trainable while leaving the rest frozen, significantly reducing memory usage for both storage and working memory compared to training the entire embedding layer.

This approach is particularly useful for domain adaptation, adding new vocabulary, or fine-tuning specific tokens while preserving the rest of the pre-trained embeddings.

=== Usage ===
TrainableTokensConfig is used to specify which token indices should be trainable during fine-tuning. It extends PeftConfig and provides configuration for targeting embedding layers and initializing token weights. The configuration is passed to get_peft_model() to create a model with trainable token adapters.

== Code Reference ==

=== Source Location ===
File: src/peft/tuners/trainable_tokens/config.py
Lines: 25-90

=== Class Signature ===
<syntaxhighlight lang="python">
@dataclass
class TrainableTokensConfig(PeftConfig):
    """
    Configuration for the TrainableTokens method.

    Args:
        token_indices: List of token indices to make trainable
        target_modules: Module names or regex for embedding layers
        init_weights: Initialize to existing embeddings (default: True)
    """
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.trainable_tokens.config import TrainableTokensConfig
# Or via the main PEFT interface
from peft import TrainableTokensConfig
</syntaxhighlight>

== I/O Contract ==

=== Configuration Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| token_indices || list[int] || [] || List of token indices to make trainable
|-
| target_modules || Optional[Union[list[str], str]] || None || Module names or regex for embedding layers to adapt
|-
| init_weights || bool || True || Initialize to existing embeddings; False for random initialization
|}

=== Attributes ===
{| class="wikitable"
! Attribute !! Type !! Description
|-
| peft_type || PeftType || Set to PeftType.TRAINABLE_TOKENS in __post_init__
|}

== Usage Examples ==

=== Basic Configuration for New Tokens ===
<syntaxhighlight lang="python">
from peft import TrainableTokensConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Add new tokens
new_tokens = ["<special1>", "<special2>", "<domain_term>"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# Get indices of new tokens
token_indices = [tokenizer.convert_tokens_to_ids(token) for token in new_tokens]

# Configure trainable tokens
config = TrainableTokensConfig(
    token_indices=token_indices,
    init_weights=False  # Random init for new tokens
)

# Apply configuration
model = get_peft_model(model, config)
</syntaxhighlight>

=== Re-training Specific Existing Tokens ===
<syntaxhighlight lang="python">
# Identify tokens to fine-tune (e.g., domain-specific terms)
tokens_to_finetune = ["medical", "diagnosis", "treatment", "patient"]
token_indices = [tokenizer.convert_tokens_to_ids(token) for token in tokens_to_finetune]

# Configure with initialization from existing embeddings
config = TrainableTokensConfig(
    token_indices=token_indices,
    init_weights=True  # Start from current embeddings
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Custom Target Modules ===
<syntaxhighlight lang="python">
# Target specific embedding layers
config = TrainableTokensConfig(
    token_indices=[100, 101, 102, 103],
    target_modules=["encoder.embed_tokens", "decoder.embed_tokens"]
)

model = get_peft_model(base_model, config)
</syntaxhighlight>

=== Finding Token Indices ===
<syntaxhighlight lang="python">
# Method 1: By token strings
tokens = ["hello", "world", "AI"]
token_indices = [tokenizer.convert_tokens_to_ids(t) for t in tokens]

# Method 2: By tokenizing text
text = "special domain terminology"
encoded = tokenizer(text, return_tensors="pt")
token_indices = encoded.input_ids[0].tolist()

# Method 3: Range of new tokens after resizing
original_vocab_size = len(tokenizer)
tokenizer.add_tokens(["<new1>", "<new2>", "<new3>"])
model.resize_token_embeddings(len(tokenizer))
token_indices = list(range(original_vocab_size, len(tokenizer)))

config = TrainableTokensConfig(token_indices=token_indices)
</syntaxhighlight>

=== Training with TrainableTokens ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Configure trainable tokens
config = TrainableTokensConfig(
    token_indices=[50000, 50001, 50002],  # Indices of new tokens
    init_weights=False
)

model = get_peft_model(base_model, config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./trainable_tokens_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

# Train only the specified tokens
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
</syntaxhighlight>

=== Checking Trainable Parameters ===
<syntaxhighlight lang="python">
# Print trainable parameters
model.print_trainable_parameters()

# Verify only specified tokens are trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}, Shape: {param.shape}")
</syntaxhighlight>

== Related Pages ==
* [[huggingface_peft_TrainableTokensLayer|TrainableTokensLayer]] - Layer implementation
* [[huggingface_peft_TrainableTokensModel|TrainableTokensModel]] - Model class
* [[Token Embeddings]]
* [[Parameter-Efficient Fine-Tuning (PEFT)]]
* [[Domain Adaptation]]

[[Category:Machine Learning]]
[[Category:PEFT]]
[[Category:Model Configuration]]
[[Category:NLP]]
[[Category:HuggingFace]]
