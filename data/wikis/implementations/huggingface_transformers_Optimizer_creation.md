{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Trainer Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete optimizer and learning rate scheduler factory methods provided by HuggingFace Transformers Trainer.

=== Description ===
The create_optimizer() and create_scheduler() methods implement the optimizer and scheduler setup principle by providing intelligent factory functions that automatically configure appropriate optimization components based on TrainingArguments. These methods handle parameter grouping, optimizer selection, and scheduler instantiation with sensible defaults while allowing customization.

The create_optimizer() method separates model parameters into groups for selective weight decay application, supports multiple optimizer types (AdamW, Adafactor, SGD, specialized variants), and handles edge cases like frozen parameters and layer-wise optimization. The create_scheduler() method computes training duration, calculates warmup steps, and instantiates the appropriate scheduler type with correct timing parameters.

These implementations are automatically called by the Trainer if custom optimizers/schedulers aren't provided, but can also be overridden in subclasses for custom behavior.

=== Usage ===
These methods are typically called automatically by Trainer during initialization or at the start of training. Use them directly when you need to manually recreate optimizer/scheduler (e.g., after loading from checkpoint), when implementing custom Trainer subclasses with specialized optimization logic, or when you want to inspect the default optimizer/scheduler that Trainer would create. Call create_optimizer() first, then create_scheduler() passing the total number of training steps.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py:L1203-1250 (create_optimizer), L1749-1766 (create_scheduler)

=== Signature ===
<syntaxhighlight lang="python">
def create_optimizer(self):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """

def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
        optimizer (torch.optim.Optimizer, optional): Optimizer to create scheduler for (uses self.optimizer if None).
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer
# Methods are part of Trainer class, not imported separately
</syntaxhighlight>

== I/O Contract ==

=== Inputs (create_optimizer) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| self.model || nn.Module || Yes || Model whose parameters will be optimized
|-
| self.args || TrainingArguments || Yes || Contains optimizer configuration (optim type, lr, weight_decay, etc.)
|-
| self.optimizer || Optimizer or None || Yes || If not None, method returns without creating new optimizer
|}

=== Inputs (create_scheduler) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| num_training_steps || int || Yes || Total number of training steps for scheduler duration calculation
|-
| optimizer || torch.optim.Optimizer || No || Optimizer to schedule (defaults to self.optimizer if None)
|-
| self.args || TrainingArguments || Yes || Contains scheduler configuration (type, warmup_steps, warmup_ratio)
|-
| self.lr_scheduler || Scheduler or None || Yes || If not None, method returns without creating new scheduler
|}

=== Outputs (create_optimizer) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| self.optimizer || torch.optim.Optimizer || Configured optimizer instance stored in trainer state
|}

=== Outputs (create_scheduler) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| self.lr_scheduler || LambdaLR or similar || Configured learning rate scheduler instance
|}

== Usage Examples ==

=== Automatic Optimizer and Scheduler Creation ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Configure optimizer and scheduler via TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    optim="adamw_torch"  # Specify optimizer type
)

# Trainer automatically calls create_optimizer() and create_scheduler()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Access the created optimizer and scheduler
print(f"Optimizer: {trainer.optimizer}")
print(f"Scheduler: {trainer.lr_scheduler}")

# Training uses these automatically
trainer.train()
</syntaxhighlight>

=== Custom Trainer with Manual Optimizer Creation ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
import torch

class CustomTrainer(Trainer):
    def create_optimizer(self):
        """Override to customize parameter grouping."""
        # Custom parameter grouping: different decay for different layers
        model = self.model
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        bert_params = ["bert" in n for n, _ in model.named_parameters()]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                          if "bert" in n and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate * 0.1,  # Lower LR for BERT layers
            },
            {
                "params": [p for n, p in model.named_parameters()
                          if "bert" in n and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate * 0.1,
            },
            {
                "params": [p for n, p in model.named_parameters()
                          if "bert" not in n and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,  # Higher LR for classifier
            },
            {
                "params": [p for n, p in model.named_parameters()
                          if "bert" not in n and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

training_args = TrainingArguments(output_dir="./custom-opt")
trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
</syntaxhighlight>

=== Manually Creating Optimizer and Scheduler ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
training_args = TrainingArguments(
    output_dir="./manual-opt",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=500
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

# Manually trigger optimizer creation (normally automatic)
trainer.create_optimizer()

# Calculate total training steps
total_steps = (
    len(train_dataset) //
    (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
) * training_args.num_train_epochs

# Manually trigger scheduler creation
trainer.create_scheduler(num_training_steps=total_steps)

print(f"Optimizer: {trainer.optimizer.__class__.__name__}")
print(f"Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
print(f"Scheduler: {trainer.lr_scheduler.__class__.__name__}")
</syntaxhighlight>

=== Different Optimizer Types ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Using different optimizers via TrainingArguments
configs = [
    # AdamW with PyTorch implementation
    TrainingArguments(
        output_dir="./adamw",
        optim="adamw_torch",
        learning_rate=5e-5,
        weight_decay=0.01
    ),

    # Adafactor (memory-efficient)
    TrainingArguments(
        output_dir="./adafactor",
        optim="adafactor",
        learning_rate=5e-5
    ),

    # 8-bit AdamW (bitsandbytes, very memory-efficient)
    TrainingArguments(
        output_dir="./adamw-8bit",
        optim="adamw_8bit",
        learning_rate=5e-5,
        weight_decay=0.01
    ),

    # SGD with momentum
    TrainingArguments(
        output_dir="./sgd",
        optim="sgd",
        learning_rate=1e-3,
        adam_beta1=0.9  # Used as momentum for SGD
    ),
]

# Each creates appropriate optimizer when Trainer initializes
for config in configs:
    trainer = Trainer(model=model, args=config, train_dataset=train_dataset)
    print(f"Optimizer: {trainer.optimizer.__class__.__name__}")
</syntaxhighlight>

=== Different Scheduler Types ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Various scheduler configurations
scheduler_configs = [
    # Linear decay with warmup (default)
    TrainingArguments(
        output_dir="./linear",
        lr_scheduler_type="linear",
        warmup_ratio=0.1
    ),

    # Cosine annealing with warmup
    TrainingArguments(
        output_dir="./cosine",
        lr_scheduler_type="cosine",
        warmup_steps=1000
    ),

    # Constant with warmup
    TrainingArguments(
        output_dir="./constant",
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05
    ),

    # Polynomial decay
    TrainingArguments(
        output_dir="./polynomial",
        lr_scheduler_type="polynomial",
        warmup_steps=500,
        lr_scheduler_kwargs={"power": 2.0}
    ),
]

for config in scheduler_configs:
    trainer = Trainer(model=model, args=config, train_dataset=train_dataset)
    print(f"Scheduler: {config.lr_scheduler_type}")
</syntaxhighlight>

=== Inspecting Parameter Groups ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments(
    output_dir="./inspect",
    learning_rate=5e-5,
    weight_decay=0.01
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

# Inspect how parameters are grouped
for i, group in enumerate(trainer.optimizer.param_groups):
    print(f"\nParameter Group {i}:")
    print(f"  Learning Rate: {group['lr']}")
    print(f"  Weight Decay: {group['weight_decay']}")
    print(f"  Number of parameters: {len(group['params'])}")

    # Show sample parameter names (requires mapping back to named_parameters)
    param_ids = {id(p) for p in group['params']}
    sample_names = [name for name, p in model.named_parameters() if id(p) in param_ids][:3]
    print(f"  Sample parameters: {sample_names}")
</syntaxhighlight>

=== Resuming with Optimizer State ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Initial training
training_args = TrainingArguments(
    output_dir="./checkpoints",
    save_strategy="steps",
    save_steps=1000
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()

# Resume from checkpoint - optimizer and scheduler states are restored
training_args_resume = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=5  # Continue for more epochs
)

trainer_resumed = Trainer(
    model=model,
    args=training_args_resume,
    train_dataset=train_dataset
)

# Optimizer and scheduler are recreated, then state is loaded from checkpoint
trainer_resumed.train(resume_from_checkpoint="./checkpoints/checkpoint-1000")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Optimizer_Scheduler_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Training_Environment]]
