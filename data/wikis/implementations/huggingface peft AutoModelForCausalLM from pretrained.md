{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for loading pre-trained transformer models from HuggingFace Hub or local paths, serving as the base model for LoRA fine-tuning.

=== Description ===

`AutoModelForCausalLM.from_pretrained` is the primary entry point for loading causal language models from the HuggingFace ecosystem. In the context of PEFT/LoRA fine-tuning, this function loads the base model that will be wrapped with adapter layers. It handles automatic model architecture detection, weight loading, and device placement.

=== Usage ===

Use this when beginning a LoRA fine-tuning workflow. This is the first step before applying any PEFT configuration. The loaded model will serve as the frozen base for adapter training. Choose appropriate `torch_dtype` and `device_map` based on your hardware constraints.

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/huggingface/transformers transformers]
* '''Class:''' `transformers.AutoModelForCausalLM`

=== Signature ===
<syntaxhighlight lang="python">
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    *model_args,
    config: Optional[PretrainedConfig] = None,
    cache_dir: Optional[str] = None,
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: Optional[bool] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = None,
    attn_implementation: Optional[str] = None,
    **kwargs
) -> PreTrainedModel:
    """
    Load a pre-trained model from HuggingFace Hub or local directory.

    Args:
        pretrained_model_name_or_path: HuggingFace Hub model ID or local path
        torch_dtype: Model precision (torch.float16, torch.bfloat16, torch.float32)
        device_map: Device placement strategy ("auto", "cuda", "cpu", or dict)
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")

    Returns:
        PreTrainedModel: Loaded model ready for adaptation
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pretrained_model_name_or_path || str || Yes || HuggingFace Hub ID (e.g., "meta-llama/Llama-2-7b-hf") or local path
|-
| torch_dtype || torch.dtype || No || Model precision. Use torch.float16 or torch.bfloat16 for memory efficiency
|-
| device_map || str or dict || No || Device placement. "auto" for automatic multi-GPU, "cuda" for single GPU
|-
| attn_implementation || str || No || Attention backend. "flash_attention_2" for speed, "sdpa" for PyTorch native
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Loaded transformer model ready for PEFT wrapping
|}

== Usage Examples ==

=== Basic Loading for LoRA ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model with half precision for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load tokenizer (typically paired with model loading)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token  # Required for some models
</syntaxhighlight>

=== Loading with Flash Attention ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM

# Load with Flash Attention 2 for faster training
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Base_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
