{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Paper|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
End-to-end process for parameter-efficient fine-tuning of large language models using 4-bit quantization (QLoRA) with Unsloth's 2x speedup optimizations.

=== Description ===
This workflow outlines the standard procedure for fine-tuning Large Language Models on consumer hardware. It combines 4-bit quantization (Q) with Low-Rank Adaptation (LoRA) to reduce memory requirements by 75%+, enabling training of 7B+ parameter models on single GPUs with 16GB VRAM. Unsloth's optimizations provide an additional 2x speedup and 30% memory reduction compared to standard QLoRA implementations.

=== Usage ===
Execute this workflow when you have an instruction dataset and need to adapt a base LLM to follow specific instructions, but have limited GPU resources (e.g., <24GB VRAM). This is the most common fine-tuning workflow for practitioners without access to large GPU clusters.

== Execution Steps ==
=== Step 1: Model Loading with Quantization ===
[[step::Principle:Quantization]]
Load the base model with 4-bit NF4 quantization to fit in GPU memory. Unsloth provides pre-quantized models for faster loading.

<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
</syntaxhighlight>

=== Step 2: LoRA Adapter Configuration ===
[[step::Principle:Low_Rank_Adaptation]]
Add LoRA adapters to the model with Unsloth's gradient checkpointing optimizations.

<syntaxhighlight lang="python">
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

=== Step 3: Dataset Preparation ===
[[step::Principle:Supervised_Fine_Tuning]]
Format your dataset into the required instruction format for the model.

<syntaxhighlight lang="python">
from datasets import load_dataset

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Optional: Apply chat template
def format_prompt(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

dataset = dataset.map(format_prompt)
</syntaxhighlight>

=== Step 4: Training Configuration ===
[[step::Principle:Supervised_Fine_Tuning]]
Configure training hyperparameters using SFTConfig.

<syntaxhighlight lang="python">
from trl import SFTConfig

args = SFTConfig(
    output_dir = "outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 10,
    max_steps = 60,
    learning_rate = 2e-4,
    optim = "adamw_8bit",
    max_seq_length = 2048,
    logging_steps = 1,
    seed = 3407,
)
</syntaxhighlight>

=== Step 5: Training Execution ===
[[step::Principle:Supervised_Fine_Tuning]]
Run the training loop using SFTTrainer.

<syntaxhighlight lang="python">
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = args,
)

trainer.train()
</syntaxhighlight>

=== Step 6: Model Saving ===
Save the trained adapter or merged model.

<syntaxhighlight lang="python">
# Save LoRA adapter only (~100MB)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Or save merged 16-bit model
model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")

# Or export to GGUF for local inference
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
</syntaxhighlight>

== Execution Diagram ==
{{#mermaid:graph TD
    A[Load Model 4-bit] --> B[Add LoRA Adapters]
    B --> C[Prepare Dataset]
    C --> D[Configure Training]
    D --> E[Train with SFTTrainer]
    E --> F{Save Model}
    F --> G[LoRA Adapter]
    F --> H[Merged Model]
    F --> I[GGUF Export]
}}

== Related Pages ==
=== Execution Steps ===
* [[step::Principle:Quantization]] - Step 1
* [[step::Principle:Low_Rank_Adaptation]] - Step 2
* [[step::Principle:Supervised_Fine_Tuning]] - Steps 3-5

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Unsloth_Gradient_Checkpointing_Optimization]]
* [[uses_heuristic::Heuristic:QLoRA_Target_Modules_Selection]]
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]
* [[uses_heuristic::Heuristic:LoRA_Rank_Selection]]
* [[uses_heuristic::Heuristic:Batch_Size_Optimization]]
* [[uses_heuristic::Heuristic:AdamW_8bit_Optimizer_Usage]]

