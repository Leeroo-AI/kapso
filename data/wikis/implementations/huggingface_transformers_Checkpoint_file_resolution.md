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
Concrete tool for discovering and resolving checkpoint file locations across formats and storage backends provided by HuggingFace Transformers.

=== Description ===
The _get_resolved_checkpoint_files function is the internal workhorse for locating model weight files in the HuggingFace Transformers library. It implements a sophisticated priority system that first attempts to find safetensors files (for security), then falls back to PyTorch pickle files if needed. The function handles both single-file and sharded checkpoints, manages GGUF quantized formats, and integrates with the HuggingFace Hub's caching system. It resolves files from local directories, cached downloads, or initiates downloads from remote repositories while respecting user preferences for format and variants.

The implementation includes automatic format conversion triggers, where it can initiate background threads to convert PyTorch checkpoints to safer safetensors format when appropriate.

=== Usage ===
This function is typically called internally by PreTrainedModel.from_pretrained() during the model loading process. It is invoked after configuration resolution and before weight loading. While not usually called directly by end users, it can be used in advanced scenarios where you need to determine checkpoint file locations without actually loading the weights.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/modeling_utils.py
* '''Lines:''' 512-785

=== Signature ===
<syntaxhighlight lang="python">
def _get_resolved_checkpoint_files(
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    variant: Optional[str],
    gguf_file: Optional[str],
    use_safetensors: Optional[bool],
    download_kwargs: DownloadKwargs,
    user_agent: dict,
    is_remote_code: bool,
    transformers_explicit_filename: Optional[str] = None,
) -> tuple[Optional[list[str]], Optional[dict]]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoModel
# Note: This function is internal and used automatically by from_pretrained
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pretrained_model_name_or_path || str or os.PathLike || No || Model identifier or path to model files
|-
| variant || str || No || Model variant suffix (e.g., "fp16", "bf16") for weight precision variants
|-
| gguf_file || str || No || Specific GGUF quantized file to load (overrides normal checkpoint discovery)
|-
| use_safetensors || bool || No || Whether to use safetensors format (None: auto, True: require, False: avoid)
|-
| download_kwargs || DownloadKwargs || Yes || Dictionary with cache_dir, token, revision, local_files_only, etc.
|-
| user_agent || dict || Yes || User agent information for tracking downloads
|-
| is_remote_code || bool || Yes || Whether the model uses remote/custom code
|-
| transformers_explicit_filename || str || No || Explicit filename from config to override discovery
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| checkpoint_files || list[str] or None || List of resolved checkpoint file paths (may contain multiple files for sharded models)
|-
| sharded_metadata || dict or None || Metadata dictionary from index file if model is sharded, containing parameter-to-shard mapping
|}

== Usage Examples ==

=== Standard Model Loading (Automatic Discovery) ===
<syntaxhighlight lang="python">
from transformers import AutoModel

# Checkpoint discovery happens automatically
# It will prefer safetensors if available, fall back to PyTorch format
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

# Force safetensors format (will error if not available)
model = AutoModel.from_pretrained(
    "google-bert/bert-base-uncased",
    use_safetensors=True
)

# Avoid safetensors, use PyTorch pickle format
model = AutoModel.from_pretrained(
    "google-bert/bert-base-uncased",
    use_safetensors=False
)
</syntaxhighlight>

=== Loading Model Variants ===
<syntaxhighlight lang="python">
from transformers import AutoModel

# Load fp16 variant (looks for model.fp16.safetensors or pytorch_model.fp16.bin)
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    variant="fp16",
    torch_dtype="auto"
)

# Load bfloat16 variant
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    variant="bf16",
    torch_dtype="auto"
)
</syntaxhighlight>

=== Loading Sharded Models ===
<syntaxhighlight lang="python">
from transformers import AutoModel

# For large models, checkpoint discovery automatically handles sharding
# Discovers model.safetensors.index.json and all corresponding shards
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto"  # Needed for large models
)

# The function discovers all files:
# - model.safetensors.index.json (metadata)
# - model-00001-of-00015.safetensors
# - model-00002-of-00015.safetensors
# - ... (all 15 shards)
</syntaxhighlight>

=== Loading GGUF Quantized Models ===
<syntaxhighlight lang="python">
from transformers import AutoModel

# Load specific GGUF quantized file
model = AutoModel.from_pretrained(
    "TheBloke/Llama-2-7B-GGUF",
    gguf_file="llama-2-7b.Q4_K_M.gguf"
)
</syntaxhighlight>

=== Local and Cached Loading ===
<syntaxhighlight lang="python">
from transformers import AutoModel

# Load from local directory (discovery checks local files first)
model = AutoModel.from_pretrained("./my_local_model")

# Load from cache only (no download attempts)
model = AutoModel.from_pretrained(
    "google-bert/bert-base-uncased",
    local_files_only=True,
    cache_dir="~/.cache/huggingface"
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Checkpoint_Discovery]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Loading_Environment]]
