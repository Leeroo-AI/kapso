# Implementation: SFTTrainer_train

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Repo|TRL|https://github.com/huggingface/trl]]
|-
! Domains
| [[domain::NLP]], [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for executing supervised fine-tuning provided by TRL and optimized by Unsloth.

=== Description ===

`SFTTrainer` (from TRL) handles the complete training loop for supervised fine-tuning. Unsloth patches it via `UnslothTrainer` to provide:

* Optimized gradient accumulation handling
* Automatic padding-free training detection
* Sequence packing support
* NEFTune noise injection integration
* Custom optimizer support for embedding learning rates

The trainer manages batching, gradient computation, optimizer steps, checkpointing, and logging throughout training.

=== Usage ===

Import SFTTrainer from TRL after importing Unsloth (to apply patches). Create a trainer instance with your model, tokenizer, dataset, and configuration, then call `.train()` to start the training loop.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/trainer.py
* '''Lines:''' 182-438 (UnslothTrainer extends SFTTrainer)

=== Signature ===
<syntaxhighlight lang="python">
# From TRL (patched by Unsloth)
class SFTTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[PreTrainedTokenizer] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,  # Deprecated, use processing_class
        packing: bool = False,
        formatting_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ):
        """
        Trainer for supervised fine-tuning.

        Args:
            model: Model with LoRA adapters from get_peft_model
            args: SFTConfig with training hyperparameters
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            processing_class: Tokenizer for processing
            packing: Enable sequence packing
            formatting_func: Custom formatting function for dataset
            compute_metrics: Function to compute evaluation metrics
            callbacks: List of TrainerCallback instances
        """

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        **kwargs,
    ) -> TrainOutput:
        """
        Execute the training loop.

        Args:
            resume_from_checkpoint: Path to checkpoint or True for latest

        Returns:
            TrainOutput with training metrics
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel  # Import first to apply patches
from trl import SFTTrainer, SFTConfig

# Or use Unsloth's trainer directly
from unsloth import UnslothTrainer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PeftModelForCausalLM || Yes || Model with LoRA adapters from get_peft_model
|-
| args || SFTConfig || Yes || Training configuration
|-
| train_dataset || Dataset || Yes || Training dataset
|-
| processing_class || PreTrainedTokenizer || Yes || Tokenizer for data processing
|-
| eval_dataset || Dataset || No || Evaluation dataset for metrics
|-
| data_collator || DataCollator || No || Custom data collator
|-
| packing || bool || No || Enable sequence packing (default: False)
|-
| formatting_func || Callable || No || Custom dataset formatting function
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| TrainOutput || namedtuple || Contains global_step, training_loss, metrics
|-
| model || (side effect) || Model weights updated in-place
|-
| checkpoints || (files) || Saved to args.output_dir every args.save_steps
|-
| logs || (files/stdout) || Training metrics logged per args.logging_steps
|}

== Usage Examples ==

=== Basic SFT Training ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load model and add LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# Load dataset
dataset = load_dataset("json", data_files="train.json", split="train")

# Configure training
training_args = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_steps=100,
)

# Create trainer and train
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
</syntaxhighlight>

=== With Sequence Packing ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

dataset = load_dataset("json", data_files="train.json", split="train")

# Enable packing in config
training_args = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_seq_length=2048,
    packing=True,  # Pack multiple samples per sequence
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
</syntaxhighlight>

=== With Custom Formatting Function ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

dataset = load_dataset("json", data_files="train.json", split="train")

# Custom formatting function
def formatting_func(examples):
    texts = []
    for prompt, response in zip(examples["prompt"], examples["response"]):
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{response}<|im_end|>"
        texts.append(text)
    return {"text": texts}

training_args = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)

trainer.train()
</syntaxhighlight>

=== Resume from Checkpoint ===
<syntaxhighlight lang="python">
# Resume training from a checkpoint
trainer.train(resume_from_checkpoint="./outputs/checkpoint-500")

# Or auto-detect latest checkpoint
trainer.train(resume_from_checkpoint=True)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_SFT_Training]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_TRL]]
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
