{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for loading pretrained model configurations from HuggingFace Hub or local filesystem provided by the HuggingFace Transformers library.

=== Description ===

`AutoConfig.from_pretrained()` is a class method of the `PreTrainedConfig` base class that instantiates a model configuration object from a pretrained checkpoint. It handles downloading configuration files from the HuggingFace Hub, caching them locally, and parsing the JSON configuration into the appropriate config class instance. This is typically the first step in the model loading workflow, as the configuration contains essential metadata about model architecture, hyperparameters, and quantization settings.

=== Usage ===

Use this when you need to:
* Load a model configuration before instantiating the model itself
* Inspect model hyperparameters without loading full model weights
* Override specific configuration attributes during model initialization
* Work with custom models hosted on HuggingFace Hub

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/configuration_utils.py (lines 494-593)

=== Signature ===
<syntaxhighlight lang="python">
@classmethod
def from_pretrained(
    cls: type[SpecificPreTrainedConfigType],
    pretrained_model_name_or_path: str | os.PathLike,
    cache_dir: str | os.PathLike | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    revision: str = "main",
    **kwargs,
) -> SpecificPreTrainedConfigType:
    """
    Instantiate a PreTrainedConfig (or a derived class) from a pretrained model configuration.

    Args:
        pretrained_model_name_or_path: Model id on huggingface.co, path to directory,
            or path to configuration JSON file
        cache_dir: Directory for caching downloaded configurations
        force_download: Force re-download even if cached
        local_files_only: Only use local files, no downloads
        token: HuggingFace Hub authentication token
        revision: Model version (branch, tag, or commit id)
        **kwargs: Additional config attributes to override

    Returns:
        PreTrainedConfig: Configuration object instantiated from pretrained model
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pretrained_model_name_or_path || str | os.PathLike || Yes || Model ID on HuggingFace Hub, local directory path, or config JSON file path
|-
| cache_dir || str | os.PathLike | None || No || Directory to cache downloaded configurations
|-
| force_download || bool || No || Whether to force re-download if cached version exists (default: False)
|-
| local_files_only || bool || No || If True, only load from local files without attempting downloads (default: False)
|-
| token || str | bool | None || No || Authentication token for private models on HuggingFace Hub
|-
| revision || str || No || Specific model version - branch name, tag, or commit id (default: "main")
|-
| **kwargs || dict || No || Additional parameters to override in the loaded configuration
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || PreTrainedConfig subclass || Configuration object (e.g., BertConfig, GPT2Config) containing model architecture parameters
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import AutoConfig

# Load configuration from HuggingFace Hub
config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
print(config.hidden_size)  # 768
print(config.num_hidden_layers)  # 12

# Load from local directory
config = AutoConfig.from_pretrained("./my_model_directory/")

# Load from specific JSON file
config = AutoConfig.from_pretrained("./my_model_directory/config.json")

# Override configuration attributes
config = AutoConfig.from_pretrained(
    "google-bert/bert-base-uncased",
    output_attentions=True,
    output_hidden_states=True
)

# Load private model with authentication
config = AutoConfig.from_pretrained(
    "my-org/private-model",
    token="hf_your_token_here"
)

# Use specific model revision
config = AutoConfig.from_pretrained(
    "google-bert/bert-base-uncased",
    revision="v1.0.0"
)

# Offline mode - only use cached files
config = AutoConfig.from_pretrained(
    "google-bert/bert-base-uncased",
    local_files_only=True
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Configuration_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
