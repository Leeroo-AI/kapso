{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Generation|https://huggingface.co/docs/transformers/main_classes/text_generation]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Inference]], [[domain::Text_Generation]], [[domain::Model_Serving]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for running text generation through a PEFT model using the transformers generation API.

=== Description ===

`model.generate()` is inherited from transformers' `GenerationMixin` and works transparently with PEFT models. The adapter weights are applied during each forward pass of the generation loop, modifying outputs according to the task-specific fine-tuning.

=== Usage ===

Use for autoregressive text generation with PEFT models. Supports all standard transformers generation parameters including sampling, beam search, and constraints.

== Code Reference ==

=== Source Location ===
* '''Library:''' `transformers.GenerationMixin` (external)
* '''Method:''' `generate()`

=== Signature ===
<syntaxhighlight lang="python">
def generate(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    max_new_tokens: int = None,
    max_length: int = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    **kwargs
) -> torch.Tensor:
    """
    Generate sequences from input_ids.

    Args:
        input_ids: Tokenized input tensor
        max_new_tokens: Maximum tokens to generate
        do_sample: Use sampling vs greedy decoding
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling
        num_beams: Beam search width

    Returns:
        Generated token IDs tensor
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# No import needed - method inherited by PeftModel
from peft import PeftModel
# model.generate() is available on PeftModel instance
</syntaxhighlight>

== Usage Examples ==

=== Greedy Generation ===
<syntaxhighlight lang="python">
from peft import PeftModel
import torch

model = PeftModel.from_pretrained(base_model, "path/to/adapter")
model.eval()

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=False,  # Greedy decoding
    )
</syntaxhighlight>

=== Sampling with Temperature ===
<syntaxhighlight lang="python">
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Inference_Execution]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
