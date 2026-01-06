{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Tokenizers|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Loading a pretrained tokenizer from a model repository or local directory with all necessary vocabulary files and configuration.

=== Description ===
Tokenizer loading is the initialization process for obtaining a ready-to-use tokenizer instance from pretrained resources. This principle addresses the need to instantiate tokenizers with consistent vocabulary, special tokens, and configuration that match the original model training setup. It fits at the beginning of the tokenization pipeline, before any text processing occurs.

The loading process involves resolving vocabulary files (either from HuggingFace Hub or local paths), loading tokenizer configuration, handling special tokens, and initializing the appropriate tokenizer class (fast or slow). This ensures that text will be tokenized exactly as it was during model training, which is critical for model performance.

=== Usage ===
This principle should be applied when:
* Starting a new text processing task with a pretrained model
* Initializing tokenizers for inference or fine-tuning
* Loading tokenizers from HuggingFace Hub model repositories
* Restoring tokenizers from locally saved directories
* Switching between different pretrained models that require different vocabularies

== Theoretical Basis ==
The tokenizer loading principle follows these logical steps:

1. '''Resource Resolution''': Identify the source of vocabulary files
   * If model identifier: resolve to HuggingFace Hub repository
   * If local path: verify directory contains required files
   * If single file: validate it's a supported format

2. '''File Discovery''': Locate all necessary tokenizer files
   * Vocabulary file(s): vocab.txt, vocab.json, merges.txt, sentencepiece.model, etc.
   * Tokenizer config: tokenizer_config.json
   * Special tokens map: special_tokens_map.json
   * Fast tokenizer file: tokenizer.json (optional, for fast tokenizers)
   * Added tokens: added_tokens.json

3. '''Configuration Loading''': Parse configuration files
   * Load tokenizer_config.json for initialization parameters
   * Extract special token definitions
   * Determine tokenizer type and backend

4. '''Vocabulary Initialization''': Load the core vocabulary
   * Parse vocabulary files according to tokenizer type
   * Build token-to-id and id-to-token mappings
   * Initialize subword tokenization algorithm state

5. '''Special Token Setup''': Configure special tokens
   * Load predefined special tokens (PAD, UNK, CLS, SEP, MASK, BOS, EOS)
   * Register added tokens with their IDs
   * Preserve special token ordering and properties

6. '''Instance Creation''': Instantiate the tokenizer class
   * Select appropriate tokenizer class (Fast vs Slow)
   * Initialize with loaded vocabulary and configuration
   * Apply any user-provided overrides

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_PreTrainedTokenizerBase_from_pretrained]]
