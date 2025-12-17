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

Concrete tool for discovering and downloading all checkpoint shard files for large sharded models provided by the HuggingFace Transformers library.

=== Description ===

`get_checkpoint_shard_files()` is a utility function that handles the complexity of working with sharded model checkpoints. Large language models often have weights split across multiple files (shards) for easier storage and distribution. This function reads an index file (typically `model.safetensors.index.json` or `pytorch_model.bin.index.json`) that maps parameter names to their corresponding shard files, then downloads and caches all necessary shards. It returns the complete list of local file paths along with metadata about the checkpoint structure.

=== Usage ===

Use this when you need to:
* Load large models that are split across multiple checkpoint files
* Download all shards of a model from HuggingFace Hub efficiently
* Work with model loading pipelines that require knowledge of all shard locations
* Implement custom weight loading logic that operates on individual shards

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/utils/hub.py (lines 827-883)

=== Signature ===
<syntaxhighlight lang="python">
def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    local_files_only=False,
    token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    _commit_hash=None,
    **deprecated_kwargs,
):
    """
    For a given model:
    - download and cache all the shards of a sharded checkpoint if
      pretrained_model_name_or_path is a model ID on the Hub
    - returns the list of paths to all the shards, as well as some metadata.

    Args:
        pretrained_model_name_or_path: Model ID on Hub or local directory path
        index_filename: Full path to the index file (downloaded and cached)
        cache_dir: Directory for caching downloaded files
        force_download: Force re-download even if cached
        proxies: Dictionary of proxy servers to use
        local_files_only: Only use local files, no downloads
        token: HuggingFace Hub authentication token
        user_agent: User agent string for HTTP requests
        revision: Model version (branch, tag, or commit id)
        subfolder: Subfolder within model repo containing the files
        _commit_hash: Commit hash for cache optimization

    Returns:
        tuple: (shard_filenames, sharded_metadata)
            - shard_filenames: List of paths to all checkpoint shard files
            - sharded_metadata: Dict containing 'all_checkpoint_keys', 'weight_map',
              and other metadata from the index
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers.utils.hub import get_checkpoint_shard_files
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pretrained_model_name_or_path || str || Yes || Model ID on HuggingFace Hub or local directory path
|-
| index_filename || str || Yes || Full path to the downloaded/cached index file (e.g., model.safetensors.index.json)
|-
| cache_dir || str | None || No || Directory for caching downloaded checkpoint files
|-
| force_download || bool || No || Whether to force re-download if cached version exists (default: False)
|-
| proxies || dict | None || No || Dictionary of proxy servers by protocol or endpoint
|-
| local_files_only || bool || No || If True, only use local files without attempting downloads (default: False)
|-
| token || str | None || No || Authentication token for private models on HuggingFace Hub
|-
| user_agent || str | dict | None || No || User agent string for HTTP requests
|-
| revision || str | None || No || Specific model version - branch name, tag, or commit id
|-
| subfolder || str || No || Subfolder within model repo containing the checkpoint files (default: "")
|-
| _commit_hash || str | None || No || Internal parameter for cache optimization
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| shard_filenames || list[str] || List of paths to all checkpoint shard files (local paths after download/caching)
|-
| sharded_metadata || dict || Metadata dictionary containing 'all_checkpoint_keys' (list of parameter names), 'weight_map' (mapping from parameter names to shard files), and other metadata from the index file
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers.utils.hub import get_checkpoint_shard_files, cached_file
import json

# First, download and cache the index file
model_id = "meta-llama/Llama-2-70b-hf"
index_filename = cached_file(
    model_id,
    "model.safetensors.index.json",
    cache_dir="~/.cache/huggingface"
)

# Now get all shard files
shard_files, metadata = get_checkpoint_shard_files(
    pretrained_model_name_or_path=model_id,
    index_filename=index_filename,
    cache_dir="~/.cache/huggingface"
)

print(f"Number of shards: {len(shard_files)}")
print(f"Total parameters: {len(metadata['all_checkpoint_keys'])}")
print(f"First shard: {shard_files[0]}")

# Check which shard contains a specific parameter
param_name = "model.layers.0.self_attn.q_proj.weight"
shard_containing_param = metadata['weight_map'][param_name]
print(f"{param_name} is in {shard_containing_param}")

# Load from local directory
local_model_path = "./my_sharded_model"
local_index = f"{local_model_path}/model.safetensors.index.json"
local_shards, local_metadata = get_checkpoint_shard_files(
    pretrained_model_name_or_path=local_model_path,
    index_filename=local_index
)

# Use with authentication for private models
private_model = "my-org/private-70b-model"
private_index = cached_file(
    private_model,
    "model.safetensors.index.json",
    token="hf_your_token_here"
)
private_shards, _ = get_checkpoint_shard_files(
    pretrained_model_name_or_path=private_model,
    index_filename=private_index,
    token="hf_your_token_here"
)

# Offline mode - only use cached files
offline_shards, _ = get_checkpoint_shard_files(
    pretrained_model_name_or_path=model_id,
    index_filename=index_filename,
    local_files_only=True
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Checkpoint_Discovery]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
