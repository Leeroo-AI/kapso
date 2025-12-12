{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Doc|PEFT Loading|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::Checkpointing]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
End-to-end process for resuming training from a previously saved LoRA adapter checkpoint with Unsloth.

=== Description ===
This workflow enables continuing training from a saved checkpoint, either to extend training, fine-tune on additional data, or recover from interruptions. Unsloth supports loading saved LoRA adapters and resuming training seamlessly. Essential for iterative model improvement and long training runs.

=== Usage ===
Execute this workflow when you need to resume interrupted training, extend training for more steps/epochs, or fine-tune a previously trained adapter on new data. Also useful for curriculum learning approaches where training progresses through multiple stages.

== Execution Steps ==
=== Step 1: Load Base Model ===
[[step::Principle:Quantization]]
Load the same base model used in original training.

<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
</syntaxhighlight>

=== Step 2: Load Saved LoRA Adapter ===
[[step::Principle:Low_Rank_Adaptation]]
Load the previously trained LoRA weights.

<syntaxhighlight lang="python">
from peft import PeftModel

model = PeftModel.from_pretrained(
    model,
    "path/to/saved/lora_adapter",
    is_trainable = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

=== Step 3: Prepare New/Additional Dataset ===
[[step::Principle:Supervised_Fine_Tuning]]
Load dataset for continued training.

<syntaxhighlight lang="python">
from datasets import load_dataset

new_dataset = load_dataset("your_new_dataset", split="train")

def format_prompt(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

new_dataset = new_dataset.map(format_prompt)
</syntaxhighlight>

=== Step 4: Configure Training ===
Set up training arguments, potentially with modified hyperparameters.

<syntaxhighlight lang="python">
from trl import SFTConfig

args = SFTConfig(
    output_dir = "continued_training_outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 100,
    learning_rate = 1e-4,
    optim = "adamw_8bit",
    max_seq_length = 2048,
    logging_steps = 1,
)
</syntaxhighlight>

=== Step 5: Resume Training ===
Continue training from checkpoint.

<syntaxhighlight lang="python">
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = new_dataset,
    args = args,
)

trainer.train(resume_from_checkpoint = True)
</syntaxhighlight>

=== Step 6: Save Updated Model ===
Save the updated adapter.

<syntaxhighlight lang="python">
model.save_pretrained("continued_lora_model")
tokenizer.save_pretrained("continued_lora_model")
</syntaxhighlight>

== Execution Diagram ==
{{#mermaid:graph TD
    A[Load Base Model] --> B[Load Saved LoRA Adapter]
    B --> C[Prepare New Dataset]
    C --> D[Configure Training]
    D --> E{Resume Options}
    E --> F[Extend Training]
    E --> G[New Data Fine-tuning]
    E --> H[Curriculum Stage]
    F --> I[Continue SFTTrainer]
    G --> I
    H --> I
    I --> J[Save Updated Adapter]
}}

== Alternative: Resume from Trainer Checkpoint ==
If you saved full trainer state:

<syntaxhighlight lang="python">
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = args,
)

trainer.train(resume_from_checkpoint = "outputs/checkpoint-500")
</syntaxhighlight>

== Related Pages ==
=== Execution Steps ===
* [[step::Principle:Quantization]] - Step 1
* [[step::Principle:Low_Rank_Adaptation]] - Step 2
* [[step::Principle:Supervised_Fine_Tuning]] - Steps 3-5

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Unsloth_Gradient_Checkpointing_Optimization]]
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]

