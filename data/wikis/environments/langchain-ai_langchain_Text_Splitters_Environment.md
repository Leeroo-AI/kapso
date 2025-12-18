# Text Splitters Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain Text Splitters|https://github.com/langchain-ai/langchain/tree/main/libs/text-splitters]]
* [[source::Doc|Text Splitters API|https://docs.langchain.com/oss/python/langchain/how_to/text_splitters]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::NLP]], [[domain::RAG]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Python environment with `langchain-text-splitters` package and optional NLP libraries (tiktoken, NLTK, spaCy, sentence-transformers) for advanced text chunking strategies.

=== Description ===
This environment provides the text splitting infrastructure for document processing in LangChain applications. The base `langchain-text-splitters` package requires only `langchain-core`, but specific splitter implementations have optional dependencies for token-based splitting (tiktoken), sentence boundary detection (NLTK, spaCy), and semantic chunking (sentence-transformers).

=== Usage ===
Use this environment for any **Text Splitting Workflow** including:
- **RecursiveCharacterTextSplitter** - Basic character-based splitting (no extra deps)
- **TokenTextSplitter** - Token-based splitting using tiktoken
- **NLTKTextSplitter** - Sentence-based splitting using NLTK
- **SpacyTextSplitter** - Sentence-based splitting using spaCy models
- **SentenceTransformersTokenTextSplitter** - Token alignment with embedding models

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows || Cross-platform Python
|-
| Python || 3.9+ || Required for type hints
|-
| RAM || 2GB+ || spaCy/sentence-transformers models require more memory
|-
| Disk || Variable || NLTK data: ~50MB; spaCy models: 50-500MB; sentence-transformers: 100MB-1GB
|}

== Dependencies ==

=== Core Package (Required) ===
* `langchain-text-splitters` - Base package with `RecursiveCharacterTextSplitter`
* `langchain-core` - Document and transformer abstractions

=== Optional Dependencies ===

| Feature | Package | Required For |
|---------|---------|--------------|
| Token counting (OpenAI) | `tiktoken` | `TokenTextSplitter`, `from_tiktoken_encoder()` |
| HuggingFace tokenizers | `transformers` | `from_huggingface_tokenizer()` |
| Sentence splitting (NLTK) | `nltk` | `NLTKTextSplitter` |
| Sentence splitting (spaCy) | `spacy` | `SpacyTextSplitter` |
| Semantic chunking | `sentence-transformers` | `SentenceTransformersTokenTextSplitter` |

=== spaCy Models (if using SpacyTextSplitter) ===
* Default: `en_core_web_sm` - English small model
* For faster splitting: Use `pipeline='sentencizer'` (no model download needed)

== Quick Install ==
<syntaxhighlight lang="bash">
# Core package only (RecursiveCharacterTextSplitter)
pip install langchain-text-splitters

# With tiktoken for token-based splitting
pip install langchain-text-splitters tiktoken

# With NLTK for sentence-based splitting
pip install langchain-text-splitters nltk
python -c "import nltk; nltk.download('punkt')"  # Download punkt tokenizer

# With spaCy for sentence-based splitting
pip install langchain-text-splitters spacy
python -m spacy download en_core_web_sm  # Download English model

# With sentence-transformers for semantic chunking
pip install langchain-text-splitters sentence-transformers

# Full NLP setup
pip install langchain-text-splitters tiktoken nltk spacy sentence-transformers
python -m spacy download en_core_web_sm
</syntaxhighlight>

== Code Evidence ==

Tiktoken check from `base.py:25-30`:
<syntaxhighlight lang="python">
try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False
</syntaxhighlight>

Tiktoken import error from `base.py:200-206`:
<syntaxhighlight lang="python">
if not _HAS_TIKTOKEN:
    msg = (
        "Could not import tiktoken python package. "
        "This is needed in order to calculate max_tokens_for_prompt. "
        "Please install it with `pip install tiktoken`."
    )
    raise ImportError(msg)
</syntaxhighlight>

NLTK check and error from `nltk.py:9-14, 36-38`:
<syntaxhighlight lang="python">
try:
    import nltk

    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

# In __init__:
if not _HAS_NLTK:
    msg = "NLTK is not installed, please install it with `pip install nltk`."
    raise ImportError(msg)
</syntaxhighlight>

spaCy check and error from `spacy.py:9-21, 62-64`:
<syntaxhighlight lang="python">
try:
    import spacy
    from spacy.lang.en import English

    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False

# In _make_spacy_pipeline_for_splitting:
if not _HAS_SPACY:
    msg = "Spacy is not installed, please install it with `pip install spacy`."
    raise ImportError(msg)
</syntaxhighlight>

sentence-transformers check from `sentence_transformers.py:9-17, 33-39`:
<syntaxhighlight lang="python">
try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

# In __init__:
if not _HAS_SENTENCE_TRANSFORMERS:
    msg = (
        "Could not import sentence_transformers python package. "
        "This is needed in order to use SentenceTransformersTokenTextSplitter. "
        "Please install it with `pip install sentence-transformers`."
    )
    raise ImportError(msg)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Could not import tiktoken python package` || tiktoken not installed || `pip install tiktoken`
|-
|| `NLTK is not installed` || NLTK not installed || `pip install nltk`
|-
|| `Spacy is not installed` || spaCy not installed || `pip install spacy`
|-
|| `Could not import sentence_transformers python package` || sentence-transformers not installed || `pip install sentence-transformers`
|-
|| `OSError: [E050] Can't find model 'en_core_web_sm'` || spaCy model not downloaded || `python -m spacy download en_core_web_sm`
|-
|| `Resource punkt not found` || NLTK punkt tokenizer not downloaded || `python -c "import nltk; nltk.download('punkt')"`
|-
|| `chunk_size must be > 0` || Invalid chunk_size parameter || Use positive integer for chunk_size
|-
|| `chunk_overlap must be >= 0` || Negative chunk_overlap || Use non-negative integer for chunk_overlap
|-
|| `Got a larger chunk overlap than chunk size` || chunk_overlap > chunk_size || Reduce chunk_overlap to be less than chunk_size
|}

== Compatibility Notes ==

* '''tiktoken:''' Required for accurate token counting with OpenAI models. Without it, character-based length functions are used.
* '''NLTK:''' Requires downloading the `punkt` tokenizer data after installation.
* '''spaCy:''' Default model `en_core_web_sm` must be downloaded separately. For faster operation without model download, use `pipeline='sentencizer'`.
* '''sentence-transformers:''' Downloads models on first use; requires internet access initially. Model caching available.
* '''HuggingFace tokenizers:''' The `transformers` package is checked but not explicitly imported in base.py.

== Related Pages ==
* [[requires_env::Implementation:langchain-ai_langchain_RecursiveCharacterTextSplitter]]
* [[requires_env::Implementation:langchain-ai_langchain_TextSplitter_length_functions]]
* [[requires_env::Implementation:langchain-ai_langchain_TextSplitter_init]]
* [[requires_env::Implementation:langchain-ai_langchain_TextSplitter_split_methods]]
* [[requires_env::Implementation:langchain-ai_langchain_TextSplitter_create_documents]]
