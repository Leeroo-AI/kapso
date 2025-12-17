{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for loading pre-trained tokenizers from HuggingFace Hub or local directories provided by the HuggingFace Transformers library.

=== Description ===

This method instantiates a tokenizer from a pre-trained model identifier or local path. It handles automatic tokenizer detection, vocabulary file loading, special tokens configuration, and backend selection (fast vs slow tokenizers). The method downloads necessary files from the HuggingFace Hub if not found locally.

=== Usage ===

Use this when you need to load a tokenizer for text encoding/decoding with a specific pre-trained model. Essential for text preprocessing before model inference or training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py (lines 1512-1769)
* '''Auto Class:''' src/transformers/models/auto/tokenization_auto.py

=== Signature ===
<syntaxhighlight lang="python">
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: Union[str, os.PathLike],
    *init_inputs,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    trust_remote_code: bool = False,
    **kwargs,
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pretrained_model_name_or_path || str or os.PathLike || Yes || Model ID from HuggingFace Hub or path to directory with tokenizer files
|-
| cache_dir || str or os.PathLike || No || Directory to cache downloaded tokenizer files
|-
| force_download || bool || No || Whether to re-download files even if cached (default: False)
|-
| local_files_only || bool || No || Whether to only use local files without downloading (default: False)
|-
| token || str or bool || No || HuggingFace API token for private models (True uses saved token)
|-
| revision || str || No || Specific model version (branch, tag, or commit) to use (default: "main")
|-
| trust_remote_code || bool || No || Whether to allow custom tokenizer code from Hub (default: False)
|-
| use_fast || bool || No || Whether to use fast tokenizer implementation if available
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokenizer || PreTrainedTokenizer or PreTrainedTokenizerFast || Initialized tokenizer instance ready for text encoding/decoding
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Load tokenizer from HuggingFace Hub
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load from local directory
tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")

# Load specific version with authentication
tokenizer = AutoTokenizer.from_pretrained(
    "private-org/model-name",
    token="hf_...",
    revision="v2.0"
)

# Use fast tokenizer for better performance
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    use_fast=True
)

# Override special tokens during loading
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased",
    pad_token="[PAD]",
    additional_special_tokens=["[CUSTOM]"]
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Tokenizer_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
