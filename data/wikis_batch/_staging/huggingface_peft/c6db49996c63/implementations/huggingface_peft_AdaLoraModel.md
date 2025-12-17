# Implementation: huggingface_peft_AdaLoraModel

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_peft|https://github.com/huggingface/peft]]
* [[source::Paper|AdaLoRA|https://openreview.net/forum?id=lq62uWRJjiY]]
|-
! Domains
| [[domain::NLP]], [[domain::Parameter_Efficient_Training]], [[domain::Model_Orchestration]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Model orchestration class for AdaLoRA that manages adaptive rank allocation, orthogonal regularization, and module creation for SVD-based low-rank adaptation.

=== Description ===

`AdaLoraModel` extends `LoraModel` to provide adaptive low-rank adaptation with dynamic rank pruning. It manages the `RankAllocator` for importance-based rank allocation, adds orthogonal regularization to the loss function, and handles creation of quantization-aware SVD layers (8-bit, 4-bit, GPTQ). Unlike standard LoRA, AdaLoRA supports only one trainable adapter at a time.

=== Usage ===

Use `AdaLoraModel` when you want automatic rank optimization during fine-tuning. It's ideal when the optimal rank is unknown and you want the model to learn which attention heads need more capacity. The model automatically adds orthogonal regularization loss and handles rank budget scheduling.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft huggingface_peft]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/model.py src/peft/tuners/adalora/model.py]
* '''Lines:''' 1-347

=== Signature ===
<syntaxhighlight lang="python">
class AdaLoraModel(LoraModel):
    """
    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model.

    Args:
        model: The model to be adapted (PreTrainedModel).
        config: The configuration of the AdaLora model (AdaLoraConfig).
        adapter_name: The name of the adapter, defaults to "default".
        low_cpu_mem_usage: Create empty adapter weights on meta device.

    Returns:
        torch.nn.Module: The AdaLora model.
    """
    target_module_mapping = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING

    def __init__(self, model, config, adapter_name, **kwargs): ...
    def forward(self, *args, **kwargs): ...
    def update_and_allocate(self, global_step): ...
    def resize_modules_by_rank_pattern(self, rank_pattern, adapter_name): ...
    def resize_state_dict_by_rank_pattern(self, rank_pattern, state_dict, adapter_name): ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import AdaLoraModel, AdaLoraConfig
# Or via get_peft_model
from peft import get_peft_model, AdaLoraConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs (forward) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| *args || Any || Yes || Positional arguments passed to base model forward
|-
| **kwargs || Any || No || Keyword arguments passed to base model forward
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs || ModelOutput || Model outputs with orthogonal regularization added to loss
|}

=== update_and_allocate ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| global_step || int || Yes || Current training step for budget scheduling
|}

== Usage Examples ==

=== Complete AdaLoRA Training ===
<syntaxhighlight lang="python">
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import AdaLoraConfig, get_peft_model
from torch.optim import AdamW
from datasets import load_dataset

# 1. Load model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# 2. Configure AdaLoRA
config = AdaLoraConfig(
    peft_type="ADALORA",
    task_type="SEQ_2_SEQ_LM",
    init_r=12,
    target_r=4,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
    total_step=10000,      # Total training steps
    tinit=500,             # Warmup steps
    tfinal=2000,           # Final pruning phase steps
    orth_reg_weight=0.5,   # Orthogonality regularization weight
)

# 3. Create AdaLoRA model
model = get_peft_model(model, config)

# 4. Training loop
optimizer = AdamW(model.parameters(), lr=3e-4)
dataset = load_dataset("glue", "mrpc")

for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss  # Includes orth_reg automatically

    loss.backward()
    optimizer.step()

    # Update rank allocation
    model.base_model.update_and_allocate(step)

    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
</syntaxhighlight>

=== Loading Saved AdaLoRA Checkpoint ===
<syntaxhighlight lang="python">
from peft import PeftModel, AdaLoraConfig

# Load with saved rank pattern
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
peft_model = PeftModel.from_pretrained(model, "path/to/adalora_checkpoint")

# The rank_pattern is restored from config
# Model uses the optimized rank allocation from training
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:huggingface_peft_CUDA_Training]]
