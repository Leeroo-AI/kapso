# File: `libs/text-splitters/langchain_text_splitters/nltk.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 59 |
| Classes | `NLTKTextSplitter` |
| Imports | __future__, langchain_text_splitters, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Splits text using NLTK's linguistic sentence tokenization for more accurate sentence boundary detection than regex-based approaches.

**Mechanism:** Inherits from TextSplitter and uses NLTK's Punkt tokenizer for sentence segmentation. Supports two modes: standard tokenization using nltk.tokenize.sent_tokenize() with configurable language parameter (default "english"), and span-based tokenization using _get_punkt_tokenizer().span_tokenize() when use_span_tokenize=True. Span mode preserves whitespace between sentences by including text from previous sentence end to current sentence start. Validates that separator is empty when using span tokenization. After sentence extraction, uses inherited _merge_splits() to combine sentences into chunks respecting chunk_size and overlap.

**Significance:** Provides linguistically-informed sentence splitting superior to regex patterns, especially for complex punctuation cases (abbreviations, decimals, ellipses). The Punkt tokenizer uses machine learning to identify sentence boundaries. Critical for applications requiring precise sentence segmentation in multiple languages. The span tokenization mode is particularly valuable for preserving original document formatting and whitespace.
