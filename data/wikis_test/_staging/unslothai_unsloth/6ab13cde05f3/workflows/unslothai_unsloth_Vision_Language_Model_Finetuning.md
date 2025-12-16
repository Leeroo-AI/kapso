{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Vision Fine-tuning Guide|https://docs.unsloth.ai/basics/vision-fine-tuning]]
* [[source::Blog|Vision Models Blog|https://unsloth.ai/blog/vision]]
|-
! Domains
| [[domain::Vision_Language_Models]], [[domain::Fine_Tuning]], [[domain::Multimodal]], [[domain::QLoRA]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==
End-to-end process for fine-tuning Vision-Language Models (VLMs) like Qwen2.5-VL, Llama 3.2 Vision, and LLaVA-style models using Unsloth's optimized framework.

=== Description ===
This workflow enables fine-tuning of multimodal models that process both images and text. Unsloth's VLM support allows training these large models on consumer GPUs through:

* **Unified Loading API**: `FastVisionModel.from_pretrained()` handles both vision encoder and language model loading
* **Selective LoRA Application**: Apply LoRA to vision layers, language layers, or both independently
* **Memory Optimization**: 4-bit quantization and gradient checkpointing reduce VRAM requirements dramatically
* **Image Processing**: Automatic handling of image tokens, attention masks, and multi-image inputs

Supported VLM architectures include:
* Qwen2.5-VL / Qwen3-VL
* Llama 3.2 Vision (11B)
* Pixtral
* LLaVA-style models
* Gemma 3 Vision

=== Usage ===
Execute this workflow when:
* You have image-text paired data for instruction tuning (OCR, visual QA, image captioning)
* You need to adapt a VLM to specific visual understanding tasks
* You want to fine-tune vision understanding alongside language capabilities
* You have a GPU with 16GB+ VRAM (24GB+ recommended for larger models)

Input requirements:
* Dataset with image paths/URLs and corresponding text instructions/responses
* Images in standard formats (PNG, JPG, WebP)
* GPU with CUDA capability 8.0+ recommended for best performance

Output:
* Trained LoRA adapters for both vision and language components
* Or merged model for deployment

== Execution Steps ==

=== Step 1: Import Vision Model Classes ===
[[step::Principle:unslothai_unsloth_Package_Initialization]]
Import the FastVisionModel class which provides the unified API for vision-language models.

```python
from unsloth import FastVisionModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
```

=== Step 2: Load Vision-Language Model ===
[[step::Principle:unslothai_unsloth_Vision_Model_Loading]]
Load the VLM with quantization settings. The loader automatically detects the architecture and applies appropriate optimizations.

```python
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = None,  # Auto-detect best dtype
    # trust_remote_code = True,  # For some model architectures
)
```

=== Step 3: Apply LoRA to Vision and Language Components ===
[[step::Principle:unslothai_unsloth_Vision_LoRA_Injection]]
Apply LoRA adapters with fine-grained control over which components to fine-tune.

```python
model = FastVisionModel.get_peft_model(
    model,
    r = 16,
    target_modules = "all-linear",     # Apply to all linear layers
    finetune_vision_layers = True,     # Train vision encoder
    finetune_language_layers = True,   # Train language model
    finetune_attention_modules = True, # Train attention layers
    finetune_mlp_modules = True,       # Train MLP layers
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

=== Step 4: Prepare Vision-Text Dataset ===
[[step::Principle:unslothai_unsloth_Vision_Data_Formatting]]
Format the dataset with image paths and conversation structure. The processor handles image tokenization automatically.

```python
# Load dataset with image paths and text
dataset = load_dataset("json", data_files="vision_train.json", split="train")

# Format for VLM training
def format_vision_data(examples):
    formatted = []
    for img_path, instruction, response in zip(
        examples["image"], examples["instruction"], examples["response"]
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
        ]
        formatted.append({"messages": messages})
    return {"formatted": formatted}

dataset = dataset.map(format_vision_data, batched=True)
```

=== Step 5: Configure Training with Vision Data Collator ===
[[step::Principle:unslothai_unsloth_Vision_SFT_Training]]
Set up training with the specialized VLM data collator that handles image processing.

```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    args = SFTConfig(
        max_seq_length = 2048,
        per_device_train_batch_size = 1,  # VLMs need smaller batches
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 30,
        logging_steps = 1,
        output_dir = "vision_outputs",
        optim = "adamw_8bit",
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        remove_unused_columns = False,  # Important for VLMs
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
    ),
)

trainer.train()
```

=== Step 6: Save Vision Model ===
[[step::Principle:unslothai_unsloth_Vision_Model_Saving]]
Save the trained VLM with both vision and language adapter weights.

```python
# Save LoRA adapters
model.save_pretrained("vision_lora_model")
tokenizer.save_pretrained("vision_lora_model")

# Or merge and save for deployment
model.save_pretrained_merged(
    "vision_merged_model",
    tokenizer,
    save_method = "merged_16bit",
)
```

== Execution Diagram ==
{{#mermaid:graph TD
    A[Import FastVisionModel] --> B[Load VLM with 4-bit Quantization]
    B --> C[Apply LoRA to Vision + Language]
    C --> D[Prepare Vision-Text Dataset]
    D --> E[Configure Training with VLM Collator]
    E --> F[Run Training]
    F --> G{Save Method}
    G -->|LoRA| H[Save Vision LoRA Adapters]
    G -->|Merged| I[Merge & Save Full Model]
}}

== Related Pages ==
* [[step::Principle:unslothai_unsloth_Package_Initialization]]
* [[step::Principle:unslothai_unsloth_Vision_Model_Loading]]
* [[step::Principle:unslothai_unsloth_Vision_LoRA_Injection]]
* [[step::Principle:unslothai_unsloth_Vision_Data_Formatting]]
* [[step::Principle:unslothai_unsloth_Vision_SFT_Training]]
* [[step::Principle:unslothai_unsloth_Vision_Model_Saving]]
