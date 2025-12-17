# Environment: NLP Text Processing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain Text Splitters|https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters]]
* [[source::Doc|pyproject.toml|libs/text-splitters/pyproject.toml]]
|-
! Domains
| [[domain::NLP]], [[domain::Text_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

## Overview

Optional NLP packages for advanced text splitting capabilities including token-based splitting and language-aware sentence segmentation.

### Description

This environment provides optional dependencies for advanced text splitting features in the `langchain-text-splitters` package. While basic character-based splitting requires no additional packages, token-based splitting (using tiktoken or transformers) and language-aware splitting (using spaCy or NLTK) require these optional dependencies.

### Usage

Install these packages when you need:
- **Token-accurate splitting** for embedding models or context windows
- **Sentence-level splitting** for maintaining semantic boundaries
- **Language-specific splitting** for multilingual documents
- **Korean text processing** with KoNLPy

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| Base || langchain-ai_langchain_Python_Runtime || Requires core environment
|-
| Python || 3.10-3.13 || Some packages have Python version constraints
|}

## Dependencies

### Core Package

* `langchain-text-splitters` >= 1.1.0, < 2.0.0
* `langchain-core` >= 1.2.0, < 2.0.0

### Optional NLP Packages

**Token Counting:**
* `tiktoken` >= 0.8.0, < 1.0.0 — OpenAI tokenizer for GPT models
* `transformers` >= 4.51.3, < 5.0.0 — HuggingFace tokenizers

**Sentence Segmentation:**
* `spacy` >= 3.8.7, < 4.0.0 — Industrial-strength NLP (Python < 3.14)
* `nltk` >= 3.9.1, < 4.0.0 — Natural Language Toolkit
* `sentence-transformers` >= 3.0.1, < 4.0.0 — Sentence embeddings (Python < 3.14)

**Korean Language:**
* `konlpy` — Korean NLP library (requires Java runtime)

**Scientific Computing:**
* `scipy` >= 1.7.0, < 2.0.0 (Python 3.12)
* `scipy` >= 1.14.1, < 2.0.0 (Python 3.13)
* `thinc` >= 8.3.6, < 9.0.0 — spaCy's ML backend

### spaCy Language Models

* `en_core_web_sm` — English model (download separately)

## Credentials

No API credentials required for NLP packages. All processing is done locally.

## Quick Install

```bash
# Install text splitters with token counting support
pip install langchain-text-splitters tiktoken

# Install with full NLP support
pip install langchain-text-splitters tiktoken spacy nltk transformers

# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Code Evidence

Optional import detection from `libs/text-splitters/langchain_text_splitters/base.py:25-37`:
```python
try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
```

Python version constraints from `libs/text-splitters/pyproject.toml:54-63`:
```toml
test_integration = [
    "spacy>=3.8.7,<4.0.0; python_version < \"3.14\"",
    "thinc>=8.3.6,<9.0.0",
    "nltk>=3.9.1,<4.0.0",
    "transformers>=4.51.3,<5.0.0",
    "sentence-transformers>=3.0.1,<4.0.0; python_version < \"3.14\"",
    "scipy>=1.7.0,<2.0.0; python_version >= \"3.12\" and python_version < \"3.13\"",
    "scipy>=1.14.1,<2.0.0; python_version >= \"3.13\"",
]
```

## Common Errors

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: tiktoken not found` || tiktoken not installed || `pip install tiktoken>=0.8.0`
|-
|| `OSError: [E050] Can't find model 'en_core_web_sm'` || spaCy model not downloaded || `python -m spacy download en_core_web_sm`
|-
|| `LookupError: NLTK punkt not found` || NLTK data not downloaded || `python -c "import nltk; nltk.download('punkt')"`
|-
|| `ModuleNotFoundError: konlpy` || KoNLPy not installed || `pip install konlpy` (requires Java)
|}

## Compatibility Notes

* **Python 3.14+:** spaCy and sentence-transformers not yet supported
* **spaCy Models:** Must be downloaded separately after installing spaCy
* **KoNLPy:** Requires Java JDK installation
* **Memory:** spaCy models require ~100-500MB depending on size

## Related Pages

* [[required_by::Implementation:langchain-ai_langchain_text_splitter_types]]
* [[required_by::Implementation:langchain-ai_langchain_split_text_method]]
* [[required_by::Implementation:langchain-ai_langchain_chunk_parameters]]
