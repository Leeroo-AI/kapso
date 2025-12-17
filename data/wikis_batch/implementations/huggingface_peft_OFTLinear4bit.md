# Implementation: huggingface_peft_OFTLinear4bit

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Controlling Text-to-Image Diffusion by Orthogonal Finetuning|https://arxiv.org/abs/2306.07280]]
* [[source::Library|bitsandbytes|https://github.com/TimDettmers/bitsandbytes]]
|-
! Domains
| [[domain::Quantization]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::4-bit Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
OFT implementation for 4-bit quantized linear layers using bitsandbytes NF4/FP4 quantization.

=== Description ===
Linear4bit extends OFTLayer to support 4-bit quantized base layers from bitsandbytes. It enables orthogonal fine-tuning on extremely memory-efficient 4-bit quantized weights, allowing training of very large models on limited hardware. The implementation handles the NF4 (Normal Float 4-bit) or FP4 quantization format, performing dequantization for rotation operations and re-quantization for storage.

Key features:
* Compatible with bnb.nn.Linear4bit quantized layers
* Supports NF4 and FP4 quantization types
* Automatic handling of quantization statistics
* Optional compute dtype specification (bfloat16/float16)
* Up to 8x memory reduction compared to FP32
* Defensive cloning for backpropagation stability

=== Usage ===
Use Linear4bit when working with extremely large models (7B+ parameters) that require maximum memory efficiency. This is the most memory-efficient OFT variant, enabling fine-tuning of models that would otherwise be impossible to train. Particularly useful for multi-billion parameter models on consumer GPUs or when maximizing batch sizes.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/bnb.py src/peft/tuners/oft/bnb.py]
* '''Lines:''' 210-389

=== Signature ===
<syntaxhighlight lang="python">
class Linear4bit(torch.nn.Module, OFTLayer):
    """OFT implemented in a dense layer with 4-bit quantization."""

    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        init_weights: bool = True,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        **kwargs,
    ) -> None:
        """Initialize 4-bit OFT layer"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.oft.bnb import Linear4bit
# Or automatically dispatched through:
from peft import get_peft_model, OFTConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| base_layer || bnb.nn.Linear4bit || Yes || 4-bit quantized linear layer
|-
| adapter_name || str || Yes || Name identifier for this adapter
|-
| r || int || No || Number of OFT blocks (default: 8)
|-
| oft_block_size || int || No || Block size (default: 0, auto-calculated)
|-
| module_dropout || float || No || Dropout probability (default: 0.0)
|-
| coft || bool || No || Use constrained OFT (default: False)
|-
| eps || float || No || COFT constraint (default: 6e-5)
|-
| block_share || bool || No || Share blocks (default: False)
|-
| init_weights || bool || No || Initialize to identity (default: True)
|-
| use_cayley_neumann || bool || No || Use approximation (default: False)
|-
| num_cayley_neumann_terms || int || No || Approximation terms (default: 5)
|}

=== Outputs ===
{| class="wikitable"
! Name !! Type !! Description
|-
| forward() output || torch.Tensor || Transformed features (same shape as input)
|-
| get_delta_weight() || torch.Tensor || Orthogonal rotation matrix
|}

== Core Methods ==

=== merge ===
<syntaxhighlight lang="python">
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    """
    Merge adapter weights into base 4-bit layer.

    Process:
    1. Dequantize 4-bit weights to float
    2. Apply OFT rotation
    3. Re-quantize to 4-bit with original parameters
    4. Update base layer weight

    Warning: May introduce rounding errors due to 4-bit precision.
    """
</syntaxhighlight>

=== unmerge ===
<syntaxhighlight lang="python">
def unmerge(self) -> None:
    """
    Unmerge adapter by applying inverse rotation.

    Process:
    1. Dequantize merged weights
    2. Apply inverse rotation (R^T)
    3. Re-quantize to 4-bit

    Warning: Accumulates quantization errors.
    """
</syntaxhighlight>

=== forward ===
<syntaxhighlight lang="python">
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Forward pass with 4-bit base and OFT rotation.

    Applies OFT rotation to input before 4-bit matmul.
    Uses defensive cloning for gradient stability.
    Handles dtype conversion for autocast compatibility.
    """
</syntaxhighlight>

== Usage Examples ==

=== Loading Model in 4-bit ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, OFTConfig

# Configure 4-bit quantization with NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
    bnb_4bit_use_double_quant=True,  # Quantize quantization constants
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    module_dropout=0.0,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

# Add OFT adapters (automatically uses Linear4bit)
peft_model = get_peft_model(model, oft_config)
peft_model.print_trainable_parameters()
# trainable params: ~4M || all params: 7B || trainable%: 0.06
</syntaxhighlight>

=== Fine-tuning Large Model in 4-bit ===
<syntaxhighlight lang="python">
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, OFTConfig, prepare_model_for_kbit_training
from datasets import load_dataset

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Add OFT
oft_config = OFTConfig(
    r=16,
    oft_block_size=0,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(model, oft_config)

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("imdb", split="train[:1000]")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./oft_mistral_4bit",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
)

# Train
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
peft_model.save_pretrained("./oft_mistral_adapter")
</syntaxhighlight>

=== Memory-Efficient Inference ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load trained adapter
peft_model = PeftModel.from_pretrained(model, "path/to/oft/adapter")

# Merge for faster inference (optional)
peft_model.merge_adapter()

# Generate
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = peft_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

print(tokenizer.decode(outputs[0]))
</syntaxhighlight>

=== Double Quantization ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, OFTConfig

# Use double quantization for even more memory savings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    quantization_config=bnb_config,
    device_map="auto"
)

oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, oft_config)

# Check memory usage
memory_mb = peft_model.get_memory_footprint() / 1e6
print(f"Model memory: {memory_mb:.2f} MB")
# Significantly less than 8-bit or FP16 versions
</syntaxhighlight>

=== Comparing Quantization Types ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, OFTConfig

oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# NF4 (Normal Float 4-bit) - recommended for most cases
bnb_config_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal distribution optimized
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_nf4 = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config_nf4,
    device_map="auto"
)
peft_nf4 = get_peft_model(model_nf4, oft_config)

# FP4 (Float 4-bit)
bnb_config_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Uniform distribution
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_fp4 = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config_fp4,
    device_map="auto"
)
peft_fp4 = get_peft_model(model_fp4, oft_config)

# Test forward pass
x = torch.randint(0, 50257, (1, 10)).to(model_nf4.device)

out_nf4 = peft_nf4(x).logits
out_fp4 = peft_fp4(x).logits

print(f"NF4 output shape: {out_nf4.shape}")
print(f"FP4 output shape: {out_fp4.shape}")
print(f"Output difference: {(out_nf4 - out_fp4).abs().mean():.6f}")
</syntaxhighlight>

=== Safe Merging with Error Checking ===
<syntaxhighlight lang="python">
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load model with trained adapter
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    quantization_config=bnb_config,
    device_map="auto"
)

peft_model = PeftModel.from_pretrained(model, "oft_adapter")

# Safe merge checks for NaNs
try:
    peft_model.merge_adapter(safe_merge=True)
    print("Merge successful, no NaNs detected")
except ValueError as e:
    print(f"Merge failed: {e}")
    print("Keeping adapter separate")

# Verify merged model
input_ids = torch.tensor([[1, 2, 3]]).to(model.device)
with torch.inference_mode():
    output = peft_model(input_ids)
    assert torch.isfinite(output.logits).all(), "Output contains NaN/Inf"

print("Merged model verified and working")
</syntaxhighlight>

== Implementation Details ==

=== 4-bit Quantization Format ===
Two quantization types:
* NF4 (Normal Float 4-bit): Optimized for normally distributed weights
* FP4 (Float 4-bit): Uniform quantization

NF4 typically provides better quality for neural network weights.

=== Quantization State ===
4-bit weights store:
* quant_state: Quantization parameters (scale, zero-point, dtype)
* quant_type: "nf4" or "fp4"
* compute_dtype: Dtype for matmul operations (bfloat16/float16)
* compress_statistics: Whether to quantize quantization constants

=== Double Quantization ===
When bnb_4bit_use_double_quant=True:
* Even the quantization constants are quantized
* Further reduces memory by ~0.4 bits per parameter
* Total: ~4.25 bits vs 4.65 bits per parameter

=== Dequantization Process ===
```python
weight = base_layer.weight  # Params4bit
output = dequantize_bnb_weight(weight, state=weight.quant_state)
# output is now FP16/BF16 for rotation
```

=== Defensive Cloning ===
The forward pass includes commented defensive cloning:
```python
# result = result.clone()
```
This addresses potential backpropagation issues with manipulated views in 4-bit training. May be needed with older PyTorch versions.

=== Memory Savings ===
4-bit quantization provides maximum savings:
* FP32: 4 bytes per parameter (1x baseline)
* FP16: 2 bytes per parameter (2x savings)
* Int8: 1 byte per parameter (4x savings)
* NF4: 0.5 bytes per parameter (8x savings)

With double quantization: ~8.5x savings.

=== Rounding Error Accumulation ===
4-bit has more quantization error than 8-bit:
* Single merge: minimal impact
* Multiple merge/unmerge: errors accumulate
* Recommendation: merge once for deployment, avoid repeated merge/unmerge

=== Compute Dtype Selection ===
bfloat16 vs float16:
* bfloat16: Better numerical stability, wider range
* float16: More precision in normal range
* bfloat16 recommended for large models

== Related Pages ==
* [[implements::Implementation:huggingface_peft_OFTLayer]]
* [[related_to::Implementation:huggingface_peft_OFTLinear8bitLt]]
* [[alternative_to::Implementation:huggingface_peft_OFTLinear]]
