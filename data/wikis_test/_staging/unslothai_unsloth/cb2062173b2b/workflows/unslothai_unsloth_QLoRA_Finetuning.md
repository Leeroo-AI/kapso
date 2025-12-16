{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|Fine-tuning Guide|https://docs.unsloth.ai/get-started/fine-tuning-guide]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::QLoRA]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

== Overview ==
End-to-end process for parameter-efficient fine-tuning (PEFT) of Large Language Models using QLoRA with 2x faster training and 70% less VRAM than standard approaches.

=== Description ===
This workflow outlines the standard procedure for fine-tuning LLMs on consumer hardware using Unsloth's optimized QLoRA implementation. It leverages 4-bit quantization (NF4) and Low-Rank Adapters (LoRA) to drastically reduce memory requirements, enabling training of 7B+ parameter models on single GPUs with limited VRAM.

The process covers:
1. **Model Loading**: Load pre-trained models with 4-bit/8-bit/16-bit quantization
2. **LoRA Configuration**: Apply LoRA adapters to attention and MLP layers
3. **Data Preparation**: Format datasets with chat templates (Alpaca, ChatML, etc.)
4. **Training**: Execute SFT training with optimized Triton kernels
5. **Saving/Export**: Merge adapters and export to various formats (HF, GGUF, Ollama)

Key optimizations include:
- Custom Triton kernels for attention, normalization, and activation functions
- Fused LoRA operations that combine base weights + adapters in single passes
- Unsloth gradient checkpointing for 30% less VRAM
- Sample packing to eliminate padding waste (2x speedup)

=== Usage ===
Execute this workflow when:
- You have a domain-specific dataset (instruction-tuning, conversational, or completion format)
- You need to adapt a base LLM (Llama, Mistral, Gemma, Qwen, etc.) to follow specific instructions
- You have limited GPU resources (e.g., <24GB VRAM for 7B models)
- You want faster training without accuracy loss

Input requirements:
- A HuggingFace-compatible dataset or local JSON/JSONL files
- Access to a base model (from HuggingFace Hub or local)
- NVIDIA GPU with CUDA capability >= 7.0 (or AMD ROCm)

== Execution Steps ==

=== Step 1: Environment Setup ===
[[step::Principle:unslothai_unsloth_Environment_Setup]]
Install Unsloth and verify hardware compatibility. The library auto-detects CUDA/HIP/XPU devices and applies appropriate optimizations.

```python
pip install unsloth

# Verify installation
from unsloth import FastLanguageModel
```

=== Step 2: Model Loading with Quantization ===
[[step::Principle:unslothai_unsloth_Model_Loading]]
Load the base model with 4-bit quantization using `FastLanguageModel.from_pretrained()`. This handles model detection, quantization, and optimization patches automatically.

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",  # or any HF model
    max_seq_length = 2048,  # Supports RoPE scaling internally
    load_in_4bit = True,    # 4-bit quantization for memory efficiency
    dtype = None,           # Auto-detect (bfloat16 for Ampere+)
)
```

=== Step 3: LoRA Adapter Configuration ===
[[step::Principle:unslothai_unsloth_LoRA_Configuration]]
Apply LoRA adapters to the model using `get_peft_model()`. Target attention projections (q, k, v, o) and MLP layers (gate, up, down) for comprehensive adaptation.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank (8, 16, 32, 64, 128 are common)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,  # 0 is optimized
    bias = "none",     # "none" is optimized
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
    random_state = 3407,
    max_seq_length = 2048,
)
```

=== Step 4: Dataset Preparation ===
[[step::Principle:unslothai_unsloth_Data_Formatting]]
Format the training dataset using chat templates. Unsloth supports 30+ templates (Alpaca, ChatML, Llama-3, etc.) via `get_chat_template()`.

```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

def formatting_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True)
```

=== Step 5: Training Execution ===
[[step::Principle:unslothai_unsloth_SFT_Training]]
Execute supervised fine-tuning using TRL's SFTTrainer with Unsloth optimizations. Sample packing can provide 2-5x speedup for short sequences.

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,  # or num_train_epochs = 1
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
        seed = 3407,
    ),
)

trainer.train()
```

=== Step 6: Model Saving and Export ===
[[step::Principle:unslothai_unsloth_Model_Export]]
Save the trained model by merging LoRA adapters back into base weights. Supports multiple export formats: LoRA-only, merged 16-bit, merged 4-bit, and GGUF.

```python
# Save merged 16-bit model (for vLLM/SGLang deployment)
model.save_pretrained_merged(
    "model_merged_16bit",
    tokenizer,
    save_method = "merged_16bit",
)

# Or save as GGUF for llama.cpp/Ollama
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",  # q8_0, q5_k_m, f16, etc.
)

# Push to HuggingFace Hub
model.push_to_hub_merged(
    "your-username/model-name",
    tokenizer,
    save_method = "merged_16bit",
)
```

== Execution Diagram ==
{{#mermaid:graph TD
    A[Step 1: Environment Setup] --> B[Step 2: Model Loading]
    B --> C[Step 3: LoRA Configuration]
    C --> D[Step 4: Dataset Preparation]
    D --> E[Step 5: SFT Training]
    E --> F[Step 6: Model Export]
    F --> G{Export Format?}
    G -->|HuggingFace| H[Merged 16-bit/4-bit]
    G -->|GGUF| I[llama.cpp/Ollama]
    G -->|LoRA| J[Adapter Only]
}}

== Related Pages ==
* [[step::Principle:unslothai_unsloth_Environment_Setup]]
* [[step::Principle:unslothai_unsloth_Model_Loading]]
* [[step::Principle:unslothai_unsloth_LoRA_Configuration]]
* [[step::Principle:unslothai_unsloth_Data_Formatting]]
* [[step::Principle:unslothai_unsloth_SFT_Training]]
* [[step::Principle:unslothai_unsloth_Model_Export]]
