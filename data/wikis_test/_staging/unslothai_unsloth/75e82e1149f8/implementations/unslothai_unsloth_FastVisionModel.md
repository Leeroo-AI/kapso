# Implementation: FastVisionModel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::VLMs]], [[domain::Vision_Language]], [[domain::Multimodal]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
Concrete tool for loading and fine-tuning Vision-Language Models (VLMs) with quantization support provided by the Unsloth library.

=== Description ===
`FastVisionModel` is an alias for `FastModel` that provides a specialized entry point for working with multimodal vision-language models in Unsloth. It supports:

1. **VLM Loading** - Automatic detection and loading of vision-language architectures
2. **Quantization** - 4-bit and 8-bit quantization for memory-efficient training
3. **Vision LoRA** - LoRA adapters for both vision encoder and language model components
4. **Multimodal Processing** - Integration with HuggingFace processors for image handling

Supported architectures include:
- Qwen2-VL (2B, 7B variants)
- Llama 3.2 Vision
- Pixtral
- Gemma 3 with vision
- Other `AutoModelForVision2Seq` compatible models

The class shares the same interface as `FastLanguageModel` but automatically selects vision-compatible model loaders and processors.

=== Usage ===
Import this class when you need to:
- Fine-tune vision-language models on image-text datasets
- Perform OCR, image captioning, or visual question answering tasks
- Apply LoRA to multimodal models
- Load VLMs with memory-efficient quantization

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai/unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L1257-L1258 unsloth/models/loader.py]
* '''Lines:''' 1257-1258 (alias definition)
* '''Base Implementation:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/vision.py#L316-L800 unsloth/models/vision.py]

Source Files: unsloth/models/loader.py:L1257-L1258; unsloth/models/vision.py:L316-L800

=== Signature ===
<syntaxhighlight lang="python">
class FastVisionModel(FastModel):
    """Alias for FastModel specialized for Vision-Language Models."""
    pass

# Inherits all methods from FastModel, which has:
class FastModel(FastBaseModel):
    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        device_map: str = "sequential",
        trust_remote_code: bool = False,
        use_gradient_checkpointing: str = "unsloth",
        revision: Optional[str] = None,
        fast_inference: bool = False,
        gpu_memory_utilization: float = 0.5,
        **kwargs,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a Vision-Language Model with optional quantization.

        Args:
            model_name: HuggingFace model ID or local path
            max_seq_length: Maximum sequence length
            dtype: Data type for computation
            load_in_4bit: Enable 4-bit quantization
            load_in_8bit: Enable 8-bit quantization
            load_in_16bit: Enable 16-bit mode
            full_finetuning: Enable full parameter training
            token: HuggingFace API token
            device_map: Device placement strategy
            trust_remote_code: Allow remote code execution
            use_gradient_checkpointing: Checkpointing mode
            fast_inference: Enable vLLM fast inference

        Returns:
            Tuple of (model, tokenizer/processor)
        """

    @staticmethod
    def get_peft_model(
        model: PreTrainedModel,
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: str = "none",
        use_gradient_checkpointing: str = "unsloth",
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Optional[dict] = None,
        **kwargs,
    ) -> PeftModel:
        """
        Add LoRA adapters to vision-language model.

        Args:
            model: Base VLM from from_pretrained()
            finetune_vision_layers: Apply LoRA to vision encoder
            finetune_language_layers: Apply LoRA to language model
            finetune_attention_modules: Target attention layers
            finetune_mlp_modules: Target MLP layers
            r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout rate
            bias: Bias handling mode
            use_gradient_checkpointing: Memory optimization
            use_rslora: Enable rank-stabilized LoRA

        Returns:
            VLM with LoRA adapters
        """

    @staticmethod
    def for_training(model: PreTrainedModel) -> None:
        """Enable training mode for the model."""

    @staticmethod
    def for_inference(model: PreTrainedModel) -> None:
        """Enable inference mode for the model."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs (from_pretrained) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || Yes || VLM model ID (e.g., "unsloth/Qwen2-VL-2B-Instruct")
|-
| max_seq_length || int || No || Maximum sequence length (default: 2048)
|-
| dtype || torch.dtype || No || Compute dtype (auto-detected if None)
|-
| load_in_4bit || bool || No || Enable 4-bit quantization (default: True)
|-
| load_in_8bit || bool || No || Enable 8-bit quantization (default: False)
|-
| token || str || No || HuggingFace API token
|-
| fast_inference || bool || No || Enable vLLM for Qwen2.5-VL/Gemma3
|}

=== Inputs (get_peft_model) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || VLM from from_pretrained()
|-
| finetune_vision_layers || bool || No || Train vision encoder (default: True)
|-
| finetune_language_layers || bool || No || Train language model (default: True)
|-
| finetune_attention_modules || bool || No || Target attention (default: True)
|-
| finetune_mlp_modules || bool || No || Target MLP (default: True)
|-
| r || int || No || LoRA rank (default: 16)
|-
| lora_alpha || int || No || LoRA scaling (default: 16)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel/PeftModel || Optimized VLM for training
|-
| tokenizer || PreTrainedTokenizer/Processor || Tokenizer or multimodal processor
|}

== Usage Examples ==

=== Basic Vision Model Fine-tuning ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# Step 1: Load vision-language model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2-VL-2B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Step 2: Add LoRA adapters for both vision and language
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # Train vision encoder
    finetune_language_layers=True,    # Train language model
    finetune_attention_modules=True,  # Target attention
    finetune_mlp_modules=True,        # Target MLP
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print(f"VLM ready for training")
</syntaxhighlight>

=== Training on OCR Dataset ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Load model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
)

# Add LoRA
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    r=16,
    lora_alpha=32,
)

# Prepare dataset with images
def format_sample(sample):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "OCR assistant"}]},
            {"role": "user", "content": [
                {"type": "text", "text": sample["question"]},
                {"type": "image", "image": sample["image"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ]
    }

dataset = load_dataset("your-ocr-dataset", split="train")
train_dataset = [format_sample(s) for s in dataset]

# Enable training mode
FastVisionModel.for_training(model)

# Setup trainer with vision data collator
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-4,
        output_dir="vision_output",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)

trainer.train()
</syntaxhighlight>

=== Language-Only Fine-tuning ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# Load VLM
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
)

# Only fine-tune language model, freeze vision encoder
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,     # Freeze vision encoder
    finetune_language_layers=True,    # Only train language
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
)
</syntaxhighlight>

=== Saving and Pushing to Hub ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# ... after training ...

# Save LoRA adapter
model.save_pretrained("vision_model_lora")
tokenizer.save_pretrained("vision_model_lora")

# Push merged model to Hub
model.push_to_hub_merged(
    "username/qwen2-vl-finetuned",
    tokenizer,
    token="hf_token",
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* Requires PyTorch with CUDA support
* Requires transformers >= 4.49.0 for Qwen2.5-VL/Pixtral
* Uses AutoProcessor for multimodal inputs

=== Tips and Tricks ===
* Use `finetune_vision_layers=False` to save memory if task doesn't need vision fine-tuning
* Set `remove_unused_columns=False` in SFTConfig for vision training
* Use `UnslothVisionDataCollator` for proper multimodal batching
