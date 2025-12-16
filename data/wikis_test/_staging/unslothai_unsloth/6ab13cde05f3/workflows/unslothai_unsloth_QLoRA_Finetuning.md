{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai]]
* [[source::Blog|Fine-tuning Guide|https://docs.unsloth.ai/get-started/fine-tuning-guide]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::QLoRA]], [[domain::Parameter_Efficient_Training]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==
End-to-end process for parameter-efficient fine-tuning (QLoRA) of Large Language Models using Unsloth's optimized kernels for 2x faster training with 70% less VRAM.

=== Description ===
This workflow outlines the standard procedure for fine-tuning Large Language Models on consumer hardware using Unsloth. It leverages 4-bit quantization (QLoRA) combined with Low-Rank Adapters (LoRA) to dramatically reduce memory requirements, enabling training of 7B+ parameter models on single GPUs with as little as 8GB VRAM.

The workflow covers:
* **Model Loading**: Loading pre-quantized or base models with automatic 4-bit quantization using `FastLanguageModel.from_pretrained()`
* **LoRA Injection**: Adding trainable LoRA adapters to attention and MLP layers via `FastLanguageModel.get_peft_model()`
* **Optimized Training**: Using HuggingFace's SFTTrainer with Unsloth's fused kernels for cross-entropy loss, RMS normalization, and RoPE embeddings
* **Model Saving**: Saving trained adapters or merging weights back to full precision for deployment

Key optimizations include:
* Fused LoRA operations that combine forward/backward passes
* Chunked cross-entropy loss for large vocabularies (>65K tokens)
* Padding-free sequence packing
* Automatic Flash Attention / xformers / SDPA backend selection

=== Usage ===
Execute this workflow when:
* You have a domain-specific dataset in instruction-tuning format (Alpaca, ChatML, etc.)
* You need to adapt a base LLM to follow specific instructions or produce domain-specific outputs
* You have limited GPU resources (8GB-24GB VRAM)
* You want faster training than standard HuggingFace PEFT without sacrificing accuracy

Input requirements:
* A HuggingFace-compatible dataset (JSON/CSV with instruction/response pairs)
* GPU with CUDA capability 7.0+ (RTX 20/30/40, V100, T4, A100, etc.)

Output:
* Trained LoRA adapter weights (small, ~100MB)
* Or merged 16-bit model for deployment

== Execution Steps ==

=== Step 1: Install and Import Unsloth ===
[[step::Principle:unslothai_unsloth_Package_Initialization]]
Install Unsloth and import the core classes. The package initialization automatically patches HuggingFace libraries for optimized performance.

```python
# Install
pip install unsloth

# Import - this triggers automatic patching of transformers/TRL/PEFT
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
```

=== Step 2: Load Model with 4-bit Quantization ===
[[step::Principle:unslothai_unsloth_Model_Loading]]
Load the base model with 4-bit quantization using `FastLanguageModel.from_pretrained()`. This handles automatic device mapping, quantization configuration, and attention backend selection.

```python
max_seq_length = 2048  # Supports RoPE scaling internally

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",  # Or any HF model
    max_seq_length = max_seq_length,
    load_in_4bit = True,   # 4-bit quantization for QLoRA
    load_in_8bit = False,  # Alternative: 8-bit quantization
    dtype = None,          # Auto-detect: bfloat16 if supported, else float16
    # token = "hf_...",    # Required for gated models
)
```

=== Step 3: Apply LoRA Adapters ===
[[step::Principle:unslothai_unsloth_LoRA_Injection]]
Inject LoRA adapters into the model's linear layers. Unsloth's implementation fuses the adapter operations for optimal performance.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                # LoRA rank (8-128 typical)
    target_modules = [     # Layers to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",     # MLP
    ],
    lora_alpha = 16,       # LoRA scaling factor
    lora_dropout = 0,      # 0 is optimized by Unsloth
    bias = "none",         # "none" is optimized
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,    # Rank-stabilized LoRA option
)
```

=== Step 4: Prepare Training Dataset ===
[[step::Principle:unslothai_unsloth_Data_Formatting]]
Load and format the dataset using the appropriate chat template. Unsloth supports 50+ model-specific templates via the chat_templates module.

```python
# Load dataset
dataset = load_dataset("json", data_files="train.json", split="train")

# Apply chat template formatting
def formatting_prompts_func(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        message = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        text = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

=== Step 5: Configure and Run Training ===
[[step::Principle:unslothai_unsloth_SFT_Training]]
Set up the SFTTrainer with optimized configuration. Unsloth automatically applies fused kernels for loss computation and normalization.

```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,           # Or num_train_epochs
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",     # 8-bit Adam for memory efficiency
        seed = 3407,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
    ),
)

# Train the model
trainer.train()
```

=== Step 6: Save Trained Model ===
[[step::Principle:unslothai_unsloth_Model_Saving]]
Save the trained model - either just the LoRA adapters (fastest, smallest) or merged to full precision for deployment.

```python
# Option 1: Save LoRA adapters only (~100MB)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Option 2: Merge and save to 16-bit (for llama.cpp/GGUF conversion)
model.save_pretrained_merged(
    "merged_model",
    tokenizer,
    save_method = "merged_16bit",  # Full precision
)

# Option 3: Push to HuggingFace Hub
model.push_to_hub_merged(
    "your-username/model-name",
    tokenizer,
    save_method = "merged_16bit",
    token = "hf_...",
)
```

== Execution Diagram ==
{{#mermaid:graph TD
    A[Install & Import Unsloth] --> B[Load Model with 4-bit Quantization]
    B --> C[Apply LoRA Adapters]
    C --> D[Prepare Training Dataset]
    D --> E[Configure & Run Training]
    E --> F{Save Method}
    F -->|LoRA Only| G[Save LoRA Adapters]
    F -->|Merged| H[Merge & Save 16-bit]
    F -->|Hub| I[Push to HuggingFace Hub]
}}

== Related Pages ==
* [[step::Principle:unslothai_unsloth_Package_Initialization]]
* [[step::Principle:unslothai_unsloth_Model_Loading]]
* [[step::Principle:unslothai_unsloth_LoRA_Injection]]
* [[step::Principle:unslothai_unsloth_Data_Formatting]]
* [[step::Principle:unslothai_unsloth_SFT_Training]]
* [[step::Principle:unslothai_unsloth_Model_Saving]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_LoRA_Rank_Selection]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Optimization]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Mixed_Precision_Training]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Padding_Free_Training]]
