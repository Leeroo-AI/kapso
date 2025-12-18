{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Tokenizer Documentation|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
|-
! Domains
| [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for loading pretrained tokenizers from HuggingFace Hub or local directories provided by HuggingFace Transformers.

=== Description ===
This class method implements the tokenizer loading principle by handling the complete workflow of resolving, downloading, and initializing tokenizers. It supports loading from model identifiers on HuggingFace Hub, local directories with saved tokenizer files, or single vocabulary files (for backward compatibility). The method automatically resolves vocabulary files, loads configurations, handles special tokens, and returns a fully initialized tokenizer instance ready for text processing.

=== Usage ===
Use this implementation when you need to:
* Load a tokenizer matching a pretrained model from HuggingFace Hub
* Initialize a tokenizer from a locally saved directory
* Override default special tokens or configuration parameters
* Work with private models using authentication tokens
* Use fast (Rust-based) or slow (Python-based) tokenizers
* Load tokenizers for models with trust_remote_code requirements

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py:L1512-1770

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
    trust_remote_code=False,
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
| pretrained_model_name_or_path || str or os.PathLike || Yes || Model identifier from HuggingFace Hub (e.g., "bert-base-uncased") or path to directory containing tokenizer files
|-
| cache_dir || str or os.PathLike || No || Directory to cache downloaded vocabulary files (defaults to ~/.cache/huggingface)
|-
| force_download || bool || No || Whether to re-download files even if cached (default: False)
|-
| local_files_only || bool || No || Whether to only use local files without attempting downloads (default: False)
|-
| token || str or bool || No || HuggingFace authentication token for private models (use True to load from saved credentials)
|-
| revision || str || No || Git revision (branch, tag, or commit) to use (default: "main")
|-
| trust_remote_code || bool || No || Whether to allow custom tokenizers from Hub (default: False, security risk if True)
|-
| **kwargs || dict || No || Additional parameters passed to tokenizer __init__ (e.g., special tokens, padding side)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokenizer || PreTrainedTokenizerBase || Fully initialized tokenizer instance (PreTrainedTokenizerFast or PreTrainedTokenizer subclass) ready for encoding text
|}

== Usage Examples ==

=== Example: Load from HuggingFace Hub ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load a BERT tokenizer from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
encoded = tokenizer("Hello, how are you?")
print(encoded)
# Output: {'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
</syntaxhighlight>

=== Example: Load from Local Directory ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Save a tokenizer first
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("./my_tokenizer")

# Load from local directory
local_tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")
</syntaxhighlight>

=== Example: Override Special Tokens ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load tokenizer with custom padding token
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    pad_token="<|endoftext|>"  # GPT-2 doesn't have a pad token by default
)

print(f"Pad token: {tokenizer.pad_token}")
print(f"Pad token ID: {tokenizer.pad_token_id}")
</syntaxhighlight>

=== Example: Load Private Model with Authentication ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load from private repository using authentication
tokenizer = AutoTokenizer.from_pretrained(
    "myorg/my-private-model",
    token=True,  # Uses token from `huggingface-cli login`
    revision="v1.0"  # Specific version
)
</syntaxhighlight>

=== Example: Fast vs Slow Tokenizer ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load fast (Rust-based) tokenizer by default
fast_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"Is fast: {fast_tokenizer.is_fast}")  # True

# Force slow (Python-based) tokenizer
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
print(f"Is fast: {slow_tokenizer.is_fast}")  # False
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Tokenizer_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Tokenization_Environment]]
