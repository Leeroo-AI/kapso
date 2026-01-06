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
Concrete tool for initializing tokenizer vocabulary from resolved files provided by HuggingFace Transformers.

=== Description ===
This class method implements the vocabulary initialization principle by processing resolved vocabulary files and configuration to create a fully initialized tokenizer instance. It handles parsing tokenizer_config.json for initialization parameters, loading added tokens and special tokens, processing chat templates, converting token formats, and finally instantiating the specific tokenizer class with the complete vocabulary and configuration. This is an internal method called by from_pretrained after vocabulary files have been resolved.

=== Usage ===
Use this implementation when:
* Being called internally by PreTrainedTokenizerBase.from_pretrained
* Processing already-resolved vocabulary file paths
* Initializing tokenizers with loaded configuration dictionaries
* Converting between slow and fast tokenizers with vocabulary transfer
* Handling backward compatibility for legacy tokenizer formats

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py:L1771-2050

=== Signature ===
<syntaxhighlight lang="python">
@classmethod
def _from_pretrained(
    cls,
    resolved_vocab_files,
    pretrained_model_name_or_path,
    init_configuration,
    *init_inputs,
    token=None,
    cache_dir=None,
    local_files_only=False,
    _commit_hash=None,
    _is_local=False,
    trust_remote_code=False,
    **kwargs,
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is an internal method, typically not imported directly
# Use AutoTokenizer.from_pretrained() instead
from transformers import AutoTokenizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| resolved_vocab_files || dict || Yes || Dictionary mapping file types to resolved file paths (vocab_file, merges_file, tokenizer_config_file, etc.)
|-
| pretrained_model_name_or_path || str || Yes || Original model identifier or path (used for name_or_path attribute)
|-
| init_configuration || dict || Yes || Base initialization configuration dictionary
|-
| token || str or bool || No || Authentication token for private models
|-
| cache_dir || str || No || Cache directory path
|-
| local_files_only || bool || No || Whether only local files are being used
|-
| _commit_hash || str || No || Git commit hash for versioning
|-
| _is_local || bool || No || Whether loading from local directory
|-
| trust_remote_code || bool || No || Whether custom code execution is allowed
|-
| **kwargs || dict || No || Additional initialization parameters to override configuration
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokenizer || PreTrainedTokenizerBase || Fully initialized tokenizer instance with complete vocabulary and configuration
|}

== Usage Examples ==

=== Example: Internal Usage Flow ===
<syntaxhighlight lang="python">
# This method is called internally by from_pretrained
# Users should use the public API:

from transformers import AutoTokenizer

# Public API - internally calls _from_pretrained
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# The internal flow is:
# 1. from_pretrained() resolves vocabulary files
# 2. _from_pretrained() loads vocab and creates tokenizer
# 3. Returns initialized tokenizer instance
</syntaxhighlight>

=== Example: Understanding Vocabulary Loading ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load tokenizer (internally uses _from_pretrained)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Examine loaded vocabulary
print(f"Vocabulary size: {len(tokenizer)}")
print(f"Special tokens: {tokenizer.all_special_tokens}")

# Check added tokens (tokens not in base vocabulary)
print(f"Added tokens: {tokenizer.added_tokens_decoder}")

# Verify special token IDs
print(f"PAD token: '{tokenizer.pad_token}' -> ID {tokenizer.pad_token_id}")
print(f"CLS token: '{tokenizer.cls_token}' -> ID {tokenizer.cls_token_id}")
print(f"SEP token: '{tokenizer.sep_token}' -> ID {tokenizer.sep_token_id}")
</syntaxhighlight>

=== Example: Chat Template Loading ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load tokenizer with chat template
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Chat template is loaded by _from_pretrained
if tokenizer.chat_template:
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]

    # Apply loaded chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print(formatted)
</syntaxhighlight>

=== Example: Added Tokens After Loading ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load base tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Vocabulary was initialized by _from_pretrained
print(f"Initial vocab size: {len(tokenizer)}")

# Add new tokens after initialization
new_tokens = ["<custom1>", "<custom2>"]
num_added = tokenizer.add_tokens(new_tokens)

print(f"Added {num_added} tokens")
print(f"New vocab size: {len(tokenizer)}")

# New tokens are now part of vocabulary
for token in new_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"Token '{token}' -> ID {token_id}")
</syntaxhighlight>

=== Example: Vocabulary Consistency Check ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Verify vocabulary bidirectional mapping
test_tokens = ["hello", "world", "[CLS]", "[SEP]"]

for token in test_tokens:
    # Token -> ID
    token_id = tokenizer.convert_tokens_to_ids(token)

    # ID -> Token (should match original)
    recovered_token = tokenizer.convert_ids_to_tokens(token_id)

    print(f"'{token}' -> {token_id} -> '{recovered_token}'")
    assert token == recovered_token, "Vocabulary mapping inconsistent!"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Vocabulary_Initialization]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Tokenization_Environment]]
