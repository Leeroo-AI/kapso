{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Model Loading Documentation|https://huggingface.co/docs/transformers/main_classes/model]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for resolving and loading model configurations from pretrained sources provided by HuggingFace Transformers.

=== Description ===
PreTrainedConfig.from_pretrained is the primary entry point for loading model configurations in the HuggingFace Transformers library. It handles the complete lifecycle of configuration resolution: from accepting flexible input formats (model IDs, local paths, URLs) to returning fully instantiated configuration objects. The method orchestrates cache management, file downloads from HuggingFace Hub, local file system access, and proper deserialization of JSON configuration files into strongly-typed configuration objects.

This implementation provides automatic model type detection, handles nested configurations for composite models, supports configuration overrides via kwargs, and integrates with HuggingFace's authentication and versioning systems.

=== Usage ===
Import and use PreTrainedConfig.from_pretrained when you need to load a model's configuration before or independently of loading the model weights. Common scenarios include inspecting model architecture parameters, creating models with custom configurations based on pretrained templates, or validating model compatibility before downloading large weight files.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/configuration_utils.py
* '''Lines:''' 493-598

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
) -> SpecificPreTrainedConfigType
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, BertConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pretrained_model_name_or_path || str or os.PathLike || Yes || Model identifier from HuggingFace Hub, path to directory with config file, or direct path to config JSON file
|-
| cache_dir || str or os.PathLike || No || Path to directory for caching downloaded configurations (defaults to ~/.cache/huggingface)
|-
| force_download || bool || No || Force re-download even if file exists in cache (default: False)
|-
| local_files_only || bool || No || Only look for local files, do not attempt to download (default: False)
|-
| token || str or bool || No || HuggingFace API token for accessing private/gated repositories
|-
| revision || str || No || Specific model version (branch, tag, or commit hash) (default: "main")
|-
| return_unused_kwargs || bool || No || If True, return tuple of (config, unused_kwargs) (default: False)
|-
| subfolder || str || No || Subfolder path within the repository where config is located
|-
| **kwargs || dict || No || Additional config attributes to override loaded values
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || PreTrainedConfig (or subclass) || Instantiated configuration object with all model hyperparameters
|}

== Usage Examples ==

=== Loading a Model Configuration ===
<syntaxhighlight lang="python">
from transformers import AutoConfig, BertConfig

# Load from HuggingFace Hub using model identifier
config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")

# Load from local directory
config = BertConfig.from_pretrained("./saved_model/")

# Load specific file
config = BertConfig.from_pretrained("./saved_model/my_config.json")

# Load with configuration overrides
config = AutoConfig.from_pretrained(
    "google-bert/bert-base-uncased",
    num_hidden_layers=6,  # Override default 12 layers
    hidden_dropout_prob=0.2
)

# Load specific version with authentication
config = AutoConfig.from_pretrained(
    "private-org/private-model",
    revision="v2.0",
    token="hf_xxxxx",
    cache_dir="/custom/cache/path"
)

# Load with unused kwargs handling
config, unused = BertConfig.from_pretrained(
    "google-bert/bert-base-uncased",
    output_attentions=True,
    custom_param=123,
    return_unused_kwargs=True
)
print(f"Unused parameters: {unused}")  # {'custom_param': 123}
</syntaxhighlight>

=== Inspecting Before Model Loading ===
<syntaxhighlight lang="python">
from transformers import AutoConfig

# Check model size before loading weights
config = AutoConfig.from_pretrained("facebook/opt-66b")
print(f"Model has {config.hidden_size} hidden size")
print(f"Vocabulary size: {config.vocab_size}")

# Determine if model fits your requirements
if config.num_hidden_layers > 24:
    print("Model too large, using smaller variant")
    config = AutoConfig.from_pretrained("facebook/opt-1.3b")
</syntaxhighlight>

=== Offline Usage ===
<syntaxhighlight lang="python">
from transformers import AutoConfig

# First download with internet connection
config = AutoConfig.from_pretrained(
    "google-bert/bert-base-uncased",
    cache_dir="./model_cache"
)

# Later use offline (reads from cache)
config = AutoConfig.from_pretrained(
    "google-bert/bert-base-uncased",
    cache_dir="./model_cache",
    local_files_only=True
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Configuration_Resolution]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Loading_Environment]]
