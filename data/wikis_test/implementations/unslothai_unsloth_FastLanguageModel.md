# Implementation: FastLanguageModel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::NLP]], [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
Concrete tool for loading and optimizing Large Language Models with 4-bit quantization provided by the Unsloth library.

=== Description ===
`FastLanguageModel` is the primary entry point class for working with language models in Unsloth. It provides a unified interface for:

1. **Model Loading** - Automatic architecture detection and loading from HuggingFace Hub or local paths
2. **Quantization** - 4-bit (NF4), 8-bit, 16-bit, and FP8 quantization support via bitsandbytes
3. **LoRA Injection** - Seamless PEFT/LoRA adapter configuration through `get_peft_model()`
4. **Optimization** - Automatic patching with Triton kernels for 2x faster training
5. **vLLM Integration** - Optional fast inference via vLLM engine

The class inherits from `FastLlamaModel` and dispatches to architecture-specific implementations (FastMistralModel, FastQwen2Model, FastGemmaModel, etc.) based on the loaded model's configuration.

=== Usage ===
Import this class when you need to:
- Fine-tune any supported language model (Llama, Mistral, Qwen, Gemma, etc.)
- Load models with 4-bit quantization for memory-efficient training
- Apply LoRA adapters to transformer models
- Use Unsloth's optimized training kernels

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai/unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L120-L621 unsloth/models/loader.py]
* '''Lines:''' 120-621

Source Files: unsloth/models/loader.py:L120-L621

=== Signature ===
<syntaxhighlight lang="python">
class FastLanguageModel(FastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        load_in_16bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        device_map: str = "sequential",
        rope_scaling: Optional[dict] = None,
        fix_tokenizer: bool = True,
        trust_remote_code: bool = False,
        use_gradient_checkpointing: str = "unsloth",
        resize_model_vocab: Optional[int] = None,
        revision: Optional[str] = None,
        use_exact_model_name: bool = False,
        offload_embedding: bool = False,
        float32_mixed_precision: Optional[bool] = None,
        fast_inference: bool = False,
        gpu_memory_utilization: float = 0.5,
        float8_kv_cache: bool = False,
        random_state: int = 3407,
        max_lora_rank: int = 64,
        disable_log_stats: bool = True,
        qat_scheme: Optional[str] = None,
        load_in_fp8: Union[bool, str] = False,
        unsloth_tiled_mlp: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pretrained model with optional quantization and optimizations.

        Args:
            model_name: HuggingFace model name or local path
            max_seq_length: Maximum sequence length for training
            dtype: Data type (torch.float16, torch.bfloat16, or None for auto)
            load_in_4bit: Enable 4-bit QLoRA quantization
            load_in_8bit: Enable 8-bit LoRA quantization
            load_in_16bit: Enable 16-bit LoRA (no quantization)
            full_finetuning: Enable full model fine-tuning (no LoRA)
            token: HuggingFace API token for private models
            device_map: Device placement strategy
            rope_scaling: RoPE scaling configuration
            fix_tokenizer: Apply tokenizer fixes automatically
            trust_remote_code: Allow remote code execution
            use_gradient_checkpointing: Gradient checkpointing mode ("unsloth" recommended)
            resize_model_vocab: Resize vocabulary to this size
            revision: Model revision/commit hash
            fast_inference: Enable vLLM-based fast inference
            gpu_memory_utilization: GPU memory fraction for vLLM
            max_lora_rank: Maximum LoRA rank for vLLM
            load_in_fp8: Enable FP8 quantization (True, False, or "block")

        Returns:
            Tuple of (model, tokenizer) ready for training or inference
        """

    @staticmethod
    def get_peft_model(
        model: PreTrainedModel,
        r: int = 16,
        target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: str = "none",
        use_gradient_checkpointing: str = "unsloth",
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Optional[dict] = None,
        modules_to_save: Optional[List[str]] = None,
        **kwargs,
    ) -> PeftModel:
        """
        Add LoRA adapters to the model for parameter-efficient fine-tuning.

        Args:
            model: Base model from from_pretrained()
            r: LoRA rank (8, 16, 32, 64, 128)
            target_modules: List of module names to apply LoRA to
            lora_alpha: LoRA scaling factor (typically equal to r)
            lora_dropout: Dropout probability (0 is optimized for speed)
            bias: Bias handling ("none", "all", "lora_only")
            use_gradient_checkpointing: Memory optimization mode
            random_state: Random seed for initialization
            use_rslora: Enable rank-stabilized LoRA
            modules_to_save: Additional modules to train fully

        Returns:
            Model with LoRA adapters ready for training
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs (from_pretrained) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || Yes || HuggingFace model ID or local path
|-
| max_seq_length || int || No || Maximum sequence length (default: 2048)
|-
| dtype || torch.dtype || No || Compute dtype (auto-detected if None)
|-
| load_in_4bit || bool || No || Enable 4-bit NF4 quantization (default: True)
|-
| load_in_8bit || bool || No || Enable 8-bit quantization (default: False)
|-
| token || str || No || HuggingFace API token for private models
|-
| fast_inference || bool || No || Enable vLLM fast inference (default: False)
|-
| use_gradient_checkpointing || str || No || "unsloth" for optimized checkpointing
|}

=== Inputs (get_peft_model) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model from from_pretrained()
|-
| r || int || No || LoRA rank (default: 16)
|-
| target_modules || List[str] || No || Modules to apply LoRA (default: attention + MLP)
|-
| lora_alpha || int || No || LoRA scaling factor (default: 16)
|-
| lora_dropout || float || No || Dropout rate (default: 0 for speed)
|-
| use_gradient_checkpointing || str || No || Checkpointing mode (default: "unsloth")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel/PeftModel || Optimized model ready for training
|-
| tokenizer || PreTrainedTokenizer || Tokenizer with applied fixes
|}

== Usage Examples ==

=== Basic QLoRA Fine-tuning ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Step 1: Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# Step 2: Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # 0 is optimized
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Model is now ready for training with SFTTrainer
print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
</syntaxhighlight>

=== Loading with vLLM Fast Inference ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with vLLM for fast inference
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,  # Enable vLLM
    gpu_memory_utilization=0.6,
    max_lora_rank=64,
)

# Generate text with optimized inference
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
</syntaxhighlight>

=== Full Fine-tuning (No LoRA) ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load for full fine-tuning
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-1B",
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_16bit=True,  # 16-bit for full finetuning
    full_finetuning=True,
)

# Model is ready for full parameter training
# Note: Requires significantly more GPU memory
</syntaxhighlight>

=== Training Embeddings ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Include embed_tokens and lm_head in training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"],  # Train embeddings
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* Requires PyTorch with CUDA support
* Requires bitsandbytes for 4-bit quantization
* Requires peft library for LoRA

=== Tips and Tricks ===
* Use `lora_dropout=0` for maximum speed (Unsloth optimized)
* Use `use_gradient_checkpointing="unsloth"` for memory efficiency
* For 7B models, 16GB VRAM is recommended with 4-bit quantization
