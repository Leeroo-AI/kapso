# File: `libs/text-splitters/langchain_text_splitters/spacy.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 71 |
| Classes | `SpacyTextSplitter` |
| Imports | __future__, langchain_text_splitters, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Splits text using spaCy's NLP models for linguistically-accurate sentence segmentation with support for multiple languages and models.

**Mechanism:** Inherits from TextSplitter and loads a spaCy language model via _make_spacy_pipeline_for_splitting(). Supports two modes: full NLP pipeline using a named model like "en_core_web_sm" (default) with NER and tagging excluded for performance, or lightweight "sentencizer" mode using English() with just the sentencizer pipe. Configurable max_length (default 1,000,000) controls maximum text size the model can process. split_text() iterates through tokenized sentences using .sents, optionally preserving whitespace with text_with_ws, then merges sentences into chunks with _merge_splits(). strip_whitespace parameter controls whether to strip or preserve inter-sentence whitespace.

**Significance:** Provides production-grade sentence splitting using spaCy's industrial-strength NLP models. More accurate than NLTK for complex text and supports 70+ languages through different spaCy models. The sentencizer mode offers fast rule-based splitting when full parsing isn't needed. Critical for multilingual applications and high-accuracy sentence boundary detection. The ability to preserve original whitespace formatting is valuable for maintaining document fidelity.
