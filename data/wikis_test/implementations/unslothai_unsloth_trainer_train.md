# Implementation: unslothai_unsloth_trainer_train

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Wrapper documentation for executing the training loop using TRL's SFTTrainer with Unsloth's optimized kernels.

=== Description ===

The `trainer.train()` method executes the supervised fine-tuning loop. With Unsloth patches applied, this uses:

1. **Triton-optimized cross-entropy loss**: 2x faster than PyTorch's implementation
2. **Fused RMS LayerNorm**: Single kernel for normalization
3. **Optimized RoPE embeddings**: Inplace rotations for memory efficiency
4. **Smart gradient checkpointing**: Selective recomputation balancing speed/memory

The training loop handles batching, gradient computation, optimization, and checkpointing automatically.

=== Usage ===

Call `trainer.train()` after configuring SFTTrainer. This is typically the final step before model saving. The method runs until max_steps or num_epochs is reached.

== Code Reference ==

=== Source Location ===
* '''External Library:''' TRL (transformers.Trainer base)
* '''Unsloth Patches:''' unsloth/trainer.py (L100-437)

=== Signature ===
<syntaxhighlight lang="python">
class SFTTrainer(Trainer):
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Optional[Any] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        """
        Execute the training loop.

        Args:
            resume_from_checkpoint: Path to checkpoint directory to resume from
            trial: Optuna trial for hyperparameter search
            ignore_keys_for_eval: Keys to ignore during evaluation

        Returns:
            TrainOutput containing global_step, training_loss, and metrics
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# trainer.train() is called on SFTTrainer instance
trainer = SFTTrainer(...)
output = trainer.train()
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| resume_from_checkpoint || str || No || Path to checkpoint directory to resume from
|-
| trainer config || (from init) || Yes || Training arguments set during SFTTrainer creation
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| TrainOutput.global_step || int || Total number of training steps completed
|-
| TrainOutput.training_loss || float || Average training loss
|-
| TrainOutput.metrics || Dict || Training metrics dictionary
|}

== Usage Examples ==

=== Basic Training ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Setup model, tokenizer, and dataset...
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = training_args,
)

# Execute training
trainer_stats = trainer.train()

# Check results
print(f"Training completed in {trainer_stats.global_step} steps")
print(f"Final loss: {trainer_stats.training_loss:.4f}")
</syntaxhighlight>

=== Monitoring Training ===
<syntaxhighlight lang="python">
# Enable verbose logging
training_args = SFTConfig(
    output_dir = "./outputs",
    logging_steps = 1,  # Log every step
    report_to = "tensorboard",  # or "wandb"
    ...
)

trainer = SFTTrainer(...)

# Training will log metrics every step
trainer.train()

# View logs in TensorBoard
# tensorboard --logdir ./outputs
</syntaxhighlight>

=== Resume from Checkpoint ===
<syntaxhighlight lang="python">
# If training was interrupted, resume from checkpoint
trainer.train(resume_from_checkpoint="./outputs/checkpoint-100")
</syntaxhighlight>

=== Training with Memory Tracking ===
<syntaxhighlight lang="python">
import torch

# Check memory before
print(f"Memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Train
trainer.train()

# Check memory after (should be similar due to efficient cleanup)
print(f"Memory after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Peak memory usage
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
</syntaxhighlight>

=== Full Training Loop Example ===
<syntaxhighlight lang="python">
import unsloth
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add LoRA
model = FastLanguageModel.get_peft_model(
    model, r = 16, lora_alpha = 16,
)

# 3. Load dataset (pre-formatted with "text" column)
dataset = load_dataset("your_dataset", split="train")

# 4. Configure training
training_args = SFTConfig(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    max_steps = 100,
    learning_rate = 2e-4,
    optim = "adamw_8bit",
    logging_steps = 1,
    save_steps = 50,
    seed = 3407,
)

# 5. Create trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = training_args,
)

# 6. Train!
print("Starting training...")
trainer_stats = trainer.train()

# 7. Results
print(f"\n{'='*50}")
print(f"Training Complete!")
print(f"Steps: {trainer_stats.global_step}")
print(f"Loss: {trainer_stats.training_loss:.4f}")
print(f"{'='*50}")

# 8. Save model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
</syntaxhighlight>

== Unsloth Kernel Optimizations ==

During training, these optimized kernels are automatically used:

| Kernel | Location | Speedup | Description |
|--------|----------|---------|-------------|
| Cross-entropy loss | `kernels/cross_entropy_loss.py` | 2x | Chunked softmax, fused loss |
| RMS LayerNorm | `kernels/rms_layernorm.py` | 1.5x | Fused forward + backward |
| RoPE embeddings | `kernels/rope_embedding.py` | 1.3x | Inplace rotations |
| SwiGLU activation | `kernels/swiglu.py` | 1.5x | Fused gate + up projection |

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_SFT_Training]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

