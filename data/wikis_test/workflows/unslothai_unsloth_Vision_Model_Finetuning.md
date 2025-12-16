# Workflow: Vision Model Fine-Tuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Vision Model Loader|https://github.com/unslothai/unsloth/blob/main/unsloth/models/vision.py]]
* [[source::Repo|Vision Training Test|https://github.com/unslothai/unsloth/blob/main/tests/saving/vision_models/test_push_to_hub_merged.py]]
* [[source::Repo|Vision OCR Benchmark|https://github.com/unslothai/unsloth/blob/main/tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py]]
|-
! Domains
| [[domain::VLMs]], [[domain::Vision_Language]], [[domain::Fine_Tuning]], [[domain::Multimodal]], [[domain::OCR]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
End-to-end process for fine-tuning Vision-Language Models (VLMs) using Unsloth's optimized training pipeline with multimodal data handling.

=== Description ===
This workflow covers fine-tuning of Vision-Language Models such as Qwen2-VL, Llama Vision, Pixtral, and other multimodal architectures. The process handles:

1. **VLM Loading** - Loading vision-language models with `FastVisionModel`
2. **Vision LoRA Configuration** - Configuring adapters for both vision encoder and language layers
3. **Multimodal Data Preparation** - Formatting image-text pairs with proper message structure
4. **Vision Training** - Using UnslothVisionDataCollator for efficient multimodal batching
5. **Model Export** - Saving and pushing merged vision models

Key features:
- Support for popular VLMs (Qwen2-VL, Llama 3.2 Vision, Pixtral, etc.)
- Option to fine-tune vision layers, language layers, or both
- Efficient image processing and batching
- Memory-optimized training with gradient checkpointing

=== Usage ===
Execute this workflow when:
- You have image-text paired datasets (OCR, image captioning, visual QA, document understanding)
- You need to adapt a vision-language model to specific visual tasks
- You want to improve model performance on domain-specific images
- You need efficient VLM training on limited GPU resources

**Input:** Pre-trained VLM + image-text dataset
**Output:** Fine-tuned vision-language model

== Execution Steps ==

=== Step 1: Environment Setup and Import ===
[[step::Principle:unslothai_unsloth_Vision_Language_Modeling]]

Import Unsloth vision components and dependencies.

```python
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
```

=== Step 2: Load Vision-Language Model ===
[[step::Principle:unslothai_unsloth_QLoRA_4bit_Quantization]]

Load the base VLM using `FastVisionModel.from_pretrained()`.

```python
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2-VL-2B-Instruct",  # or other VLMs
    max_seq_length = 2048,
    load_in_4bit = True,       # Enable 4-bit quantization
    load_in_8bit = False,      # Alternative: 8-bit quantization
    full_finetuning = False,   # Set True for full fine-tuning
)
```

**Supported Vision Models:**
- Qwen2-VL (2B, 7B variants)
- Llama 3.2 Vision
- Pixtral
- Other multimodal architectures via `AutoModelForVision2Seq`

=== Step 3: Configure Vision LoRA ===
[[step::Principle:unslothai_unsloth_Low_Rank_Adaptation]]

Configure LoRA adapters with options for vision and language layers.

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,     # Fine-tune vision encoder
    finetune_language_layers = True,   # Fine-tune language model
    finetune_attention_modules = True, # Attention layers
    finetune_mlp_modules = True,       # MLP layers
    r = 16,                            # LoRA rank
    lora_alpha = 32,                   # LoRA scaling
    lora_dropout = 0,                  # Dropout (0 is optimized)
    bias = "none",                     # Bias training
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,                # Rank-stabilized LoRA
    loftq_config = None,               # LoftQ initialization
)
```

=== Step 4: Prepare Multimodal Dataset ===
[[step::Principle:unslothai_unsloth_Vision_Language_Modeling]]

Format dataset with OpenAI-style messages including image content.

```python
# Load your dataset
dataset = load_dataset("your-ocr-dataset", split = "train")

def format_vision_data(sample):
    """Format sample with image and text in OAI message format."""
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an OCR assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["question"]},
                    {"type": "image", "image": sample["image"]},  # PIL Image
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ],
    }

# Use list comprehension to preserve PIL Image type
train_dataset = [format_vision_data(sample) for sample in dataset]
```

=== Step 5: Configure Vision Trainer ===
[[step::Principle:unslothai_unsloth_Supervised_Fine_Tuning]]

Set up the trainer with vision-specific data collator and configuration.

```python
# Enable training mode
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),  # Vision collator
    train_dataset = train_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,
        warmup_ratio = 0.03,
        max_steps = 100,  # or num_train_epochs = 2
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 5,
        save_strategy = "epoch",
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "checkpoints",
        report_to = "none",
        # REQUIRED for vision fine-tuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)
```

=== Step 6: Execute Vision Training ===
[[step::Principle:unslothai_unsloth_Gradient_Checkpointing]]

Run the training loop for vision model fine-tuning.

```python
# Execute training
trainer_stats = trainer.train()

print(f"Training completed!")
print(f"Training loss: {trainer_stats.training_loss:.4f}")
```

=== Step 7: Save and Deploy Vision Model ===
[[step::Principle:unslothai_unsloth_LoRA_Weight_Merging]]

Save the fine-tuned vision model locally or push to Hub.

```python
# Save LoRA adapter locally
model.save_pretrained("vision_model_adapter")
tokenizer.save_pretrained("vision_model_adapter")

# Push merged model to HuggingFace Hub
model.push_to_hub_merged(
    "username/vision-model-finetuned",
    tokenizer,
    token = "hf_token",
)

# Verify by loading
test_model, test_tokenizer = FastVisionModel.from_pretrained(
    "username/vision-model-finetuned"
)
```

== Execution Diagram ==

{{#mermaid:graph TD
    A[Import Unsloth Vision] --> B[Load VLM with FastVisionModel]
    B --> C[Configure Vision LoRA]
    C --> D[Format Multimodal Dataset]
    D --> E[Setup SFTTrainer with VisionDataCollator]
    E --> F[Execute Vision Training]
    F --> G{Save Method?}
    G -->|Local| H[save_pretrained]
    G -->|Hub| I[push_to_hub_merged]
    H --> J[Vision Model Ready]
    I --> J
}}

== Related Pages ==

=== Execution Steps ===
* [[step::Principle:unslothai_unsloth_Vision_Language_Modeling]] - Steps 1, 4: Environment and Data Preparation
* [[step::Principle:unslothai_unsloth_QLoRA_4bit_Quantization]] - Step 2: Model Loading
* [[step::Principle:unslothai_unsloth_Low_Rank_Adaptation]] - Step 3: LoRA Configuration
* [[step::Principle:unslothai_unsloth_Supervised_Fine_Tuning]] - Step 5: Training Setup
* [[step::Principle:unslothai_unsloth_Gradient_Checkpointing]] - Step 6: Training Execution
* [[step::Principle:unslothai_unsloth_LoRA_Weight_Merging]] - Step 7: Model Saving

=== Key Implementations ===
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]] - Vision model loader class
* [[implemented_by::Implementation:unslothai_unsloth_UnslothVisionDataCollator]] - Multimodal data collator
