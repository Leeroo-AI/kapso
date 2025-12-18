{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|AdaLoRA|https://openreview.net/forum?id=lq62uWRJjiY]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Adaptive_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Model class that creates AdaLoRA (Adaptive LoRA) from a pretrained transformer, enabling dynamic rank allocation based on importance scores during training.

=== Description ===

AdaLoraModel extends LoraModel to implement adaptive low-rank adaptation. It maintains a RankAllocator that dynamically adjusts the rank budget across layers during training based on gradient-weighted importance scores. The model supports only one trainable adapter at a time (other adapters must be in inference mode) and adds orthogonal regularization to the loss function to encourage orthogonality in the low-rank matrices.

=== Usage ===

Use AdaLoraModel when you want automatic rank allocation across layers during training. Unlike standard LoRA where you manually set ranks, AdaLoRA starts with an initial rank and prunes to a target rank, concentrating parameters in the most important layers. Best for scenarios where you want to minimize parameters while maximizing performance.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/model.py src/peft/tuners/adalora/model.py]
* '''Lines:''' 1-347

=== Signature ===
<syntaxhighlight lang="python">
class AdaLoraModel(LoraModel):
    """
    Creates AdaLoRA model from a pretrained transformers model.

    Args:
        model: The model to be adapted (PreTrainedModel)
        config: The configuration of the AdaLora model (AdaLoraConfig)
        adapter_name: The name of the adapter (default: "default")
        low_cpu_mem_usage: Create empty adapter weights on meta device

    Attributes:
        trainable_adapter_name: Name of the single trainable adapter
        rankallocator: RankAllocator for dynamic rank management
    """
    target_module_mapping = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, **kwargs):
        """Initialize AdaLoraModel with rank allocator."""

    def forward(self, *args, **kwargs):
        """Forward pass with orthogonal regularization loss."""

    def update_and_allocate(self, global_step: int):
        """
        Update AdaLoRA budget and mask.

        Must be called after loss.backward() and before zero_grad().
        """

    def resize_modules_by_rank_pattern(self, rank_pattern, adapter_name):
        """Resize modules according to computed rank pattern."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import AdaLoraModel, AdaLoraConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || The pretrained model to adapt
|-
| config || AdaLoraConfig || Yes || Configuration for AdaLoRA
|-
| adapter_name || str || No || Name for the adapter (default: "default")
|-
| low_cpu_mem_usage || bool || No || Use meta device for initialization
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| forward() || ModelOutput || Model output with loss including orthogonal regularization
|-
| update_and_allocate() || tuple || (budget, rank_pattern) for current step
|}

== Usage Examples ==

=== Creating AdaLoRA Model ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSeq2SeqLM
from peft import AdaLoraModel, AdaLoraConfig

# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Configure AdaLoRA
config = AdaLoraConfig(
    init_r=12,           # Start with rank 12
    target_r=4,          # Prune down to rank 4
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
    tinit=200,           # Warmup steps
    tfinal=1000,         # Final pruning step
    deltaT=10,           # Pruning interval
    orth_reg_weight=0.5, # Orthogonal regularization weight
)

# Create AdaLoRA model
model = AdaLoraModel(base_model, config, "default")
</syntaxhighlight>

=== Full Training Loop ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Custom training loop for AdaLoRA
class AdaLoraTrainer(Trainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        # Update rank allocation after backward pass
        # This is called automatically by the AdaLoRA callback
        return loss

# Or use callback approach
from peft.optimizers import AdaLoRACallback

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[AdaLoRACallback()],  # Handles update_and_allocate
)

trainer.train()
</syntaxhighlight>

=== Manual Rank Update ===
<syntaxhighlight lang="python">
# Manual training loop with explicit rank updates
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss  # Includes orthogonal regularization

    # Backward pass
    loss.backward()
    optimizer.step()

    # Critical: Update rank allocation
    # Must be called after backward, before zero_grad
    model.update_and_allocate(step)

    optimizer.zero_grad()

    if step % 100 == 0:
        # Check current rank budget
        config = model.peft_config[model.trainable_adapter_name]
        print(f"Step {step}, Rank pattern: {config.rank_pattern}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
