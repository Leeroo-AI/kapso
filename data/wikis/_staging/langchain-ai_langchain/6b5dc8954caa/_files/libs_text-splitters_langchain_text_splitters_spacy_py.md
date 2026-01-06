# File: `libs/text-splitters/langchain_text_splitters/spacy.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 71 |
| Classes | `SpacyTextSplitter` |
| Imports | __future__, langchain_text_splitters, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Sentence-based text splitter using spaCy's NLP models for linguistic sentence segmentation.

**Mechanism:** SpacyTextSplitter loads a spaCy language model (default "en_core_web_sm") or uses the lightweight "sentencizer" pipeline. Processes text through the spaCy pipeline to detect sentence boundaries via `.sents`. Offers strip_whitespace option to control whether trailing whitespace is preserved. The max_length parameter (default 1M characters) controls the maximum text size the model can process.

**Significance:** Provides industrial-strength linguistic sentence boundary detection using spaCy's trained models. More accurate than simple rule-based splitting for complex text. The sentencizer option offers faster processing when full linguistic analysis is unnecessary. Essential for applications requiring high-quality sentence segmentation across multiple languages.
