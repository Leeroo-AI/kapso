{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for loading and instantiating pretrained models for pipeline inference provided by the HuggingFace Transformers library.

=== Description ===

The `load_model` function handles model instantiation for pipelines. It accepts either a model identifier string (like "bert-base-uncased") or an already instantiated model object. When given a string, it attempts to load the model from the HuggingFace Hub or local cache using the appropriate model class. The function tries multiple model classes in sequence (from the provided model_classes tuple and from config.architectures) until one successfully loads. It includes fallback logic to retry with float32 if the requested dtype fails on the device.

=== Usage ===

Use this function when you need to:
* Load a pretrained model for pipeline inference
* Handle model loading with automatic class detection
* Implement dtype fallback for device compatibility
* Load models from HuggingFace Hub or local paths

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/base.py (lines 179-258)

=== Signature ===
<syntaxhighlight lang="python">
def load_model(
    model: Union[str, "PreTrainedModel"],
    config: AutoConfig,
    model_classes: Optional[tuple[type, ...]] = None,
    task: Optional[str] = None,
    **model_kwargs: Any,
) -> "PreTrainedModel":
    """
    Load a model.

    If `model` is instantiated, this function will just return it. Otherwise `model` is
    actually a checkpoint name and this method will try to instantiate it using `model_classes`.

    Args:
        model (str or PreTrainedModel): If str, a checkpoint name. The model to load.
        config (AutoConfig): The config associated with the model to help using the correct class
        model_classes (tuple[type], optional): A tuple of model classes.
        task (str): The task defining which pipeline will be returned.
        **model_kwargs: Additional keyword arguments passed to from_pretrained()

    Returns:
        PreTrainedModel: The loaded model instance.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import load_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str or PreTrainedModel || Yes || Model identifier or instantiated model object
|-
| config || AutoConfig || Yes || Model configuration containing architecture information
|-
| model_classes || tuple[type, ...] || No || Tuple of model classes to try for instantiation
|-
| task || str || No || Task name for pipeline context (added to model_kwargs)
|-
| **model_kwargs || Any || No || Additional arguments passed to from_pretrained() (device_map, dtype, etc.)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Loaded and initialized model instance ready for inference
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import AutoConfig
from transformers.pipelines.base import load_model
from transformers import BertForSequenceClassification

# Load model from checkpoint string
config = AutoConfig.from_pretrained("bert-base-uncased")
model = load_model(
    model="bert-base-uncased",
    config=config,
    model_classes=(BertForSequenceClassification,),
    task="text-classification"
)

print(type(model))  # <class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>
</syntaxhighlight>

=== Advanced Usage with Device and Dtype ===
<syntaxhighlight lang="python">
from transformers import AutoConfig
from transformers.pipelines.base import load_model
from transformers import AutoModelForCausalLM
import torch

# Load model with specific device and dtype
config = AutoConfig.from_pretrained("gpt2")
model = load_model(
    model="gpt2",
    config=config,
    model_classes=(AutoModelForCausalLM,),
    task="text-generation",
    device_map="auto",
    dtype=torch.float16
)

print(f"Model dtype: {model.dtype}")
print(f"Model device: {model.device}")
</syntaxhighlight>

=== Handling Already Instantiated Models ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, AutoModel
from transformers.pipelines.base import load_model

# Create model instance
config = AutoConfig.from_pretrained("bert-base-uncased")
model_instance = AutoModel.from_pretrained("bert-base-uncased")

# Pass already instantiated model (will be returned as-is)
result = load_model(
    model=model_instance,
    config=config,
    task="feature-extraction"
)

assert result is model_instance  # Same object returned
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Component_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]

=== Used By ===
* [[used_by::Implementation:huggingface_transformers_pipeline_factory]]
