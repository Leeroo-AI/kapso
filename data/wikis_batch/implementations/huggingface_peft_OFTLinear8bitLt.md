# Implementation: huggingface_peft_OFTLinear8bitLt

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|Controlling Text-to-Image Diffusion by Orthogonal Finetuning|https://arxiv.org/abs/2306.07280]]
* [[source::Library|bitsandbytes|https://github.com/TimDettmers/bitsandbytes]]
|-
! Domains
| [[domain::Quantization]], [[domain::Parameter-Efficient Fine-Tuning]], [[domain::8-bit Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
OFT implementation for 8-bit quantized linear layers using bitsandbytes Int8 quantization.

=== Description ===
Linear8bitLt extends OFTLayer to support 8-bit quantized base layers from the bitsandbytes library. It applies orthogonal fine-tuning transformations to weights stored in Int8 format, handling the complexities of quantization/dequantization during training and inference. The implementation dequantizes weights when needed for rotation operations, then re-quantizes for efficient storage and computation.

Key features:
* Compatible with bnb.nn.Linear8bitLt quantized layers
* Automatic dequantization for OFT operations
* Re-quantization after merging
* Maintains quantization state and statistics
* Reduces memory footprint while enabling fine-tuning
* Handles rounding errors during merge/unmerge

=== Usage ===
Use Linear8bitLt when fine-tuning large language models loaded in 8-bit precision. This enables training models that would otherwise not fit in GPU memory. Particularly useful for billion-parameter models on consumer GPUs. Note that merging may introduce small rounding errors due to quantization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/bnb.py src/peft/tuners/oft/bnb.py]
* '''Lines:''' 31-184

=== Signature ===
<syntaxhighlight lang="python">
class Linear8bitLt(torch.nn.Module, OFTLayer):
    """OFT implemented in a dense layer with 8-bit quantization."""

    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        **kwargs,
    ) -> None:
        """Initialize 8-bit OFT layer"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft.tuners.oft.bnb import Linear8bitLt
# Or automatically dispatched through:
from peft import get_peft_model, OFTConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Name !! Type !! Required !! Description
|-
| base_layer || bnb.nn.Linear8bitLt || Yes || 8-bit quantized linear layer
|-
| adapter_name || str || Yes || Name identifier for this adapter
|-
| r || int || No || Number of OFT blocks (default: 8)
|-
| oft_block_size || int || No || Block size (default: 0, auto-calculated)
|-
| module_dropout || float || No || Dropout probability (default: 0.0)
|-
| init_weights || bool || No || Initialize to identity (default: True)
|-
| coft || bool || No || Use constrained OFT (default: False)
|-
| eps || float || No || COFT constraint (default: 6e-5)
|-
| block_share || bool || No || Share blocks (default: False)
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
    Merge adapter weights into base 8-bit layer.

    Process:
    1. Dequantize Int8 weights to float
    2. Apply OFT rotation
    3. Re-quantize to Int8
    4. Update base layer weight

    Warning: May introduce rounding errors due to quantization.
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
    3. Re-quantize

    Warning: Accumulates rounding errors.
    """
</syntaxhighlight>

=== forward ===
<syntaxhighlight lang="python">
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Forward pass with 8-bit base and OFT rotation.

    Applies OFT rotation to input before 8-bit matmul.
    Handles dtype conversion for compatibility.
    """
</syntaxhighlight>

== Usage Examples ==

=== Loading Model in 8-bit ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, OFTConfig

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    quantization_config=bnb_config,
    device_map="auto"
)

# Configure OFT
oft_config = OFTConfig(
    r=8,
    module_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# Add OFT adapters (automatically uses Linear8bitLt)
peft_model = get_peft_model(model, oft_config)
peft_model.print_trainable_parameters()
# trainable params: ~2M || all params: 6.7B || trainable%: 0.03
</syntaxhighlight>

=== Fine-tuning in 8-bit ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, OFTConfig, TaskType
from datasets import load_dataset

# Load model in 8-bit
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2-large",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

# Add OFT
oft_config = OFTConfig(
    r=8,
    oft_block_size=0,
    target_modules=["c_attn"],
    task_type=TaskType.CAUSAL_LM
)
peft_model = get_peft_model(model, oft_config)

# Prepare data
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Training loop (simplified)
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

peft_model.train()
for batch in tokenized:
    input_ids = torch.tensor([batch["input_ids"]]).to(model.device)

    outputs = peft_model(input_ids, labels=input_ids)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Loss: {loss.item():.4f}")
</syntaxhighlight>

=== Merging 8-bit Adapter ===
<syntaxhighlight lang="python">
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load base model in 8-bit
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load trained adapter
peft_model = PeftModel.from_pretrained(model, "path/to/oft/adapter")

print("Before merge - using adapter:")
print(f"Merged: {peft_model.base_model.model.model.decoder.layers[0].self_attn.q_proj.merged}")

# Merge adapter into base weights
# Warning: this may introduce rounding errors
peft_model.merge_adapter()

print("\nAfter merge - adapter integrated:")
print(f"Merged: {peft_model.base_model.model.model.decoder.layers[0].self_attn.q_proj.merged}")

# Now inference uses merged 8-bit weights
input_ids = torch.tensor([[1, 2, 3]]).to(model.device)
output = peft_model(input_ids)
print(f"Output shape: {output.logits.shape}")
</syntaxhighlight>

=== Memory Comparison ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, OFTConfig

# Standard FP16 model
model_fp16 = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"FP16 model size: {model_fp16.get_memory_footprint() / 1e9:.2f} GB")

# 8-bit model with OFT
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-2.7b",
    quantization_config=bnb_config,
    device_map="auto"
)

oft_config = OFTConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
peft_model_8bit = get_peft_model(model_8bit, oft_config)

print(f"8-bit + OFT model size: {peft_model_8bit.get_memory_footprint() / 1e9:.2f} GB")
peft_model_8bit.print_trainable_parameters()
</syntaxhighlight>

=== Handling Rounding Warnings ===
<syntaxhighlight lang="python">
import warnings
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load 8-bit model with OFT
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config,
    device_map="auto"
)

peft_model = PeftModel.from_pretrained(model, "oft_adapter")

# Expect warning about rounding errors
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    peft_model.merge_adapter()

    if w:
        print(f"Warning: {w[0].message}")
        # "Merge oft module to 8-bit linear may get different generations due to rounding errors."

# Test that outputs are still reasonable
input_ids = torch.tensor([[1, 2, 3]]).to(model.device)
output_before = peft_model(input_ids, use_cache=False)

# Outputs will be slightly different from non-quantized version
# but should still be reasonable
print(f"Output logits shape: {output_before.logits.shape}")
print(f"Logits finite: {torch.isfinite(output_before.logits).all()}")
</syntaxhighlight>

== Implementation Details ==

=== Quantization Flow ===
During merge:
1. Extract Int8 weight and quantization state (SCB, statistics)
2. Call dequantize_bnb_weight() to get FP16/FP32 weights
3. Apply OFT rotation: W_new = R @ W_old
4. Create new Int8Params with re-quantized weights
5. Update base layer weight

### Dequantization Process ===
```python
weight = base_layer.weight  # Int8Params
state = base_layer.state  # QuantState
output = dequantize_bnb_weight(weight, state=state)
# output is now FP16/FP32 for rotation
```

=== State Management ===
Important state components:
* SCB: Scale and zero-point per block
* has_fp16_weights: Whether to use FP16 compute
* threshold: Outlier threshold for mixed precision
* Preserved during re-quantization

=== Rounding Errors ===
Sources of error:
1. Quantization: FP32 → Int8 loses precision
2. Dequantization: Int8 → FP32 approximate inverse
3. Re-quantization: New FP32 → Int8 loses more precision
4. Accumulation: Merge + unmerge compounds errors

Typically negligible for generation quality.

=== Autocast Handling ===
The forward pass checks for autocast:
```python
requires_conversion = not torch.is_autocast_enabled()
if requires_conversion:
    expected_dtype = x.dtype
    x = x.to(oft_R.weight.dtype)
```

Ensures compatibility with mixed precision training.

=== Memory Savings ===
8-bit quantization reduces memory by ~4x:
* FP32: 4 bytes per parameter
* FP16: 2 bytes per parameter
* Int8: 1 byte per parameter + small quantization overhead

OFT adds minimal parameters (< 1% of model size).

== Related Pages ==
* [[implements::Implementation:huggingface_peft_OFTLayer]]
* [[related_to::Implementation:huggingface_peft_OFTLinear4bit]]
* [[alternative_to::Implementation:huggingface_peft_OFTLinear]]
