# Workflow: QLoRA Fine-Tuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth QLoRA Test|https://github.com/unslothai/unsloth/blob/main/tests/qlora/test_unsloth_qlora_train_and_merge.py]]
* [[source::Repo|Unsloth GRPO Test|https://github.com/unslothai/unsloth/blob/main/tests/saving/language_models/test_save_merged_grpo_model.py]]
* [[source::Blog|Unsloth Documentation|https://unsloth.ai/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
End-to-end process for parameter-efficient fine-tuning (PEFT) of Large Language Models using 4-bit quantization (QLoRA) with Unsloth's optimized kernels for 2x faster training.

=== Description ===
This workflow outlines the standard procedure for fine-tuning Large Language Models on consumer-grade GPUs using Unsloth. It leverages 4-bit quantization via bitsandbytes (NF4) combined with Low-Rank Adapters (LoRA) to dramatically reduce memory requirements and training time. The process covers:

1. **Model Loading** - Loading pre-trained models with automatic 4-bit quantization
2. **LoRA Configuration** - Injecting trainable low-rank adapters into linear layers
3. **Dataset Preparation** - Formatting data with chat templates for instruction tuning
4. **Training** - Using SFTTrainer or GRPOTrainer with optimized Triton kernels
5. **Model Saving** - Merging LoRA weights and exporting in various formats

Key optimizations include:
- Fused attention and MLP kernels via Triton
- Optimized RMSNorm and RoPE implementations
- Gradient checkpointing with Unsloth's memory-efficient approach
- Sample packing for efficient batching

=== Usage ===
Execute this workflow when:
- You have a domain-specific dataset (instruction-tuning style, conversational, or task-specific)
- You need to adapt a base model (Llama, Mistral, Qwen, Gemma, etc.) to follow instructions or perform specific tasks
- You have limited GPU resources (works on GPUs with 8GB+ VRAM for smaller models, 16GB+ for 7B models)
- You want faster training than standard HuggingFace implementations (up to 2x speedup)

**Input:** Pre-trained model from HuggingFace Hub + formatted training dataset
**Output:** Fine-tuned LoRA adapter or merged model weights

== Execution Steps ==

=== Step 1: Environment Setup and Import ===
[[step::Principle:unslothai_unsloth_QLoRA_4bit_Quantization]]

**Critical:** Import Unsloth BEFORE other ML libraries to apply monkey-patches.

```python
# MUST import unsloth first to enable optimizations
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
```

=== Step 2: Model Loading with Quantization ===
[[step::Principle:unslothai_unsloth_QLoRA_4bit_Quantization]]

Load the base model with 4-bit quantization enabled. The `FastLanguageModel.from_pretrained()` method handles:
- Automatic model architecture detection (Llama, Mistral, Qwen, etc.)
- 4-bit quantization via bitsandbytes
- Tokenizer loading and fixing
- vLLM integration for fast inference (optional)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-1B-Instruct",  # or any supported model
    max_seq_length = 2048,
    load_in_4bit = True,        # Enable 4-bit QLoRA
    fast_inference = False,     # Set True for vLLM inference
    max_lora_rank = 64,         # Maximum LoRA rank to support
    dtype = torch.bfloat16,     # or torch.float16
)
```

=== Step 3: LoRA Adapter Configuration ===
[[step::Principle:unslothai_unsloth_Low_Rank_Adaptation]]

Inject LoRA adapters into the model's linear layers. Unsloth supports targeting attention projections and MLP layers.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,                      # LoRA rank (8, 16, 32, 64, 128)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",     # MLP
    ],
    lora_alpha = 64,             # LoRA scaling factor
    use_gradient_checkpointing = "unsloth",  # Memory optimization
    random_state = 42,
)
```

=== Step 4: Dataset Preparation ===
[[step::Principle:unslothai_unsloth_Chat_Template_Formatting]]

Format your dataset using the tokenizer's chat template for proper instruction formatting.

```python
from unsloth.chat_templates import get_chat_template

# Apply chat template (llama-3.1, alpaca, chatml, etc.)
tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        ) for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched = True)
```

=== Step 5: Training Configuration ===
[[step::Principle:unslothai_unsloth_Supervised_Fine_Tuning]]

Configure the SFTTrainer with optimized settings for QLoRA training.

```python
from unsloth import is_bfloat16_supported
from transformers import DataCollatorForSeq2Seq

training_args = SFTConfig(
    output_dir = "outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 100,              # or num_train_epochs = 1
    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",         # Memory-efficient optimizer
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 42,
    report_to = "none",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    args = training_args,
)
```

=== Step 6: Execute Training ===
[[step::Principle:unslothai_unsloth_Gradient_Checkpointing]]

Run the training loop with Unsloth's optimized kernels.

```python
# Optional: Train only on assistant responses
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Execute training
trainer_stats = trainer.train()
```

=== Step 7: Model Saving ===
[[step::Principle:unslothai_unsloth_LoRA_Weight_Merging]]

Save the trained model - either as LoRA adapter only or merged with base weights.

```python
# Option 1: Save LoRA adapter only (smallest, requires base model for inference)
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")

# Option 2: Save merged model in 16-bit (recommended for deployment)
model.save_pretrained_merged(
    "merged_16bit",
    tokenizer,
    save_method = "merged_16bit",
)

# Option 3: Push merged model to HuggingFace Hub
model.push_to_hub_merged(
    "username/model-name",
    tokenizer,
    save_method = "merged_16bit",
    token = "hf_token",
)
```

== Execution Diagram ==

{{#mermaid:graph TD
    A[Import Unsloth First] --> B[Load Model with 4-bit Quantization]
    B --> C[Configure LoRA Adapters]
    C --> D[Prepare Dataset with Chat Template]
    D --> E[Setup SFTTrainer]
    E --> F[Execute Training]
    F --> G{Save Method?}
    G -->|LoRA Only| H[save_pretrained]
    G -->|Merged| I[save_pretrained_merged]
    G -->|Hub| J[push_to_hub_merged]
}}

== Related Pages ==

=== Execution Steps ===
* [[step::Principle:unslothai_unsloth_QLoRA_4bit_Quantization]] - Steps 1-2: Environment and Model Loading
* [[step::Principle:unslothai_unsloth_Low_Rank_Adaptation]] - Step 3: LoRA Configuration
* [[step::Principle:unslothai_unsloth_Chat_Template_Formatting]] - Step 4: Data Formatting
* [[step::Principle:unslothai_unsloth_Supervised_Fine_Tuning]] - Step 5: Training Setup
* [[step::Principle:unslothai_unsloth_Gradient_Checkpointing]] - Step 6: Training Execution
* [[step::Principle:unslothai_unsloth_LoRA_Weight_Merging]] - Step 7: Model Saving

=== Key Implementations ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]] - Main model loader class
* [[implemented_by::Implementation:unslothai_unsloth_UnslothTrainer]] - Training orchestration
