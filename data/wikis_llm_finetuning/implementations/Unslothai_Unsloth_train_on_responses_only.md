# Implementation: train_on_responses_only

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL Documentation|https://huggingface.co/docs/trl]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Supervised_Learning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for masking loss computation to only assistant responses, enabling effective SFT warm-up before GRPO reinforcement learning training.

=== Description ===

`train_on_responses_only` modifies an SFTTrainer to compute loss only on assistant response tokens, masking user and system messages. This is critical for:

1. **Pre-RL SFT warm-up**: Initialize policy with good generation patterns before RL
2. **Efficient learning**: Don't waste capacity learning to predict user messages
3. **Chat format compliance**: Learn response structure without instruction repetition

=== Usage ===

Call after creating SFTTrainer but before training. Pass the delimiters that mark user vs assistant turns in your chat template.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/chat_templates.py (imported from unsloth_zoo.dataset_utils)

=== Signature ===
<syntaxhighlight lang="python">
def train_on_responses_only(
    trainer: SFTTrainer,
    instruction_part: str,
    response_part: str,
) -> SFTTrainer:
    """
    Modify trainer to compute loss only on response tokens.

    Args:
        trainer: SFTTrainer instance to modify
        instruction_part: String marking user/instruction turn start
        response_part: String marking assistant/response turn start

    Returns:
        Modified trainer (also modifies in place)
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
| trainer || SFTTrainer || Yes || Configured SFTTrainer instance
|-
| instruction_part || str || Yes || Delimiter for instruction/user turn
|-
| response_part || str || Yes || Delimiter for response/assistant turn
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| trainer || SFTTrainer || Modified trainer with response-only loss masking
|}

== Usage Examples ==

=== ChatML Template ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Apply ChatML template
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

# Create trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(...),
)

# Mask loss to responses only (ChatML delimiters)
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

trainer.train()
</syntaxhighlight>

=== Llama-3 Template ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Apply Llama-3 template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

trainer = SFTTrainer(...)

# Llama-3 specific delimiters
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer.train()
</syntaxhighlight>

=== Pre-GRPO Warm-up ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# 1. Load with vLLM for later GRPO
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 64,
)

# 2. Apply LoRA
model = FastLanguageModel.get_peft_model(model, r=64)

# 3. SFT warm-up with response-only loss
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = sft_dataset,
    args = SFTConfig(max_steps=100, per_device_train_batch_size=2),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

trainer.train()

# 4. Now ready for GRPO training
# GRPOTrainer(model=model, ...)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_SFT_Pretraining]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
