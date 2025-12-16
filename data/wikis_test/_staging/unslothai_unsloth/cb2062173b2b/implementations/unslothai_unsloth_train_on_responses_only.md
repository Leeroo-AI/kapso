{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|Train on Responses|https://docs.unsloth.ai/basics/train-on-completions-only]]
|-
! Domains
| [[domain::LLMs]], [[domain::Data_Formatting]], [[domain::Fine_Tuning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:00 GMT]]
|}

== Overview ==
Concrete tool for masking instruction tokens during training to only compute loss on assistant responses provided by the Unsloth library.

=== Description ===
`train_on_responses_only()` is a function that configures the trainer to only compute loss on assistant/response tokens, ignoring the instruction/prompt tokens. This:

1. **Masks prompt tokens** by setting their labels to -100 (ignored in cross-entropy loss)
2. **Preserves response tokens** for loss computation and gradient updates
3. **Uses instruction template markers** to identify response start/end positions
4. **Improves training efficiency** by focusing learning on the output generation task

This technique is particularly useful for:
- Instruction tuning where you only want to teach the model how to respond
- Reducing noise from prompt tokens in the loss signal
- Aligning with the actual inference task (generating responses, not prompts)

The function modifies the trainer's data collator to apply the masking at batch creation time.

=== Usage ===
Use this function when you need to:
- Train a model to generate responses without learning to repeat prompts
- Focus training signal on assistant outputs only
- Reduce training noise from instruction formatting
- Match training objective to inference task

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L40 unsloth/chat_templates.py] (re-export)
* '''Actual Implementation:''' unsloth_zoo/dataset_utils.py (external dependency)

Source Files: unsloth/chat_templates.py:L40 (re-export from unsloth_zoo.dataset_utils)

=== Signature ===
<syntaxhighlight lang="python">
def train_on_responses_only(
    trainer: SFTTrainer,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
) -> SFTTrainer:
    """
    Configure trainer to only compute loss on response tokens.

    Args:
        trainer: The SFTTrainer instance to modify
        instruction_part: String marking end of instruction (start of response)
        response_part: String marking the response (for detection)

    Returns:
        Modified trainer with response-only loss computation
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import train_on_responses_only
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| trainer || SFTTrainer || Yes || The trainer instance to modify
|-
| instruction_part || str || No || Token sequence marking instruction end
|-
| response_part || str || No || Token sequence in response section
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| trainer || SFTTrainer || Modified trainer with response-only loss
|}

== Usage Examples ==

=== Basic Usage with Llama-3 ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# Apply chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        max_steps=100,
        output_dir="outputs",
    ),
)

# Configure to train on responses only
# For Llama-3: instruction ends with <|end_header_id|> and response starts after
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    response_part="<|eot_id|>",
)

trainer.train()
</syntaxhighlight>

=== With ChatML Format ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import get_chat_template, train_on_responses_only

tokenizer = get_chat_template(tokenizer, chat_template="chatml")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# ChatML format: responses follow <|im_start|>assistant\n
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>assistant\n",
    response_part="<|im_end|>",
)
</syntaxhighlight>

=== With Alpaca Format ===
<syntaxhighlight lang="python">
from unsloth.chat_templates import get_chat_template, train_on_responses_only

tokenizer = get_chat_template(tokenizer, chat_template="alpaca")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Alpaca format: responses follow "### Response:\n"
trainer = train_on_responses_only(
    trainer,
    instruction_part="### Response:\n",
    response_part="### Instruction:",  # Next instruction marks response end
)
</syntaxhighlight>

=== Complete Training Pipeline ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# 1. Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. Apply chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# 4. Prepare dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

def format_dataset(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}

dataset = dataset.map(format_dataset)

# 5. Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="outputs",
    ),
)

# 6. Train on responses only (key step!)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    response_part="<|eot_id|>",
)

# 7. Train
trainer.train()

# 8. Save
model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Heuristic:unslothai_unsloth_Sample_Packing]]
