# File: `unsloth/dataprep/raw_text.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 348 |
| Classes | `RawTextDataLoader`, `TextPreprocessor` |
| Imports | csv, datasets, json, os, pathlib, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utilities for loading raw text from various file formats and converting it into datasets suitable for causal language model training.

**Mechanism:** The `RawTextDataLoader` class auto-detects file formats (.txt, .md, .json, .jsonl, .csv) and reads content appropriately. It tokenizes text using a provided tokenizer and performs intelligent chunking with configurable chunk size and stride overlap to maintain context across chunks. Each chunk gets an EOS token appended and is converted into a HuggingFace Dataset with `input_ids`, `attention_mask`, and `labels` fields. The `TextPreprocessor` class offers text cleaning (whitespace normalization, encoding cleanup), section extraction via regex patterns, and structure token insertion for markdown-style headers and code blocks. It also includes dataset validation to check for empty samples, encoding issues, and repeated content.

**Significance:** Core data preparation utility that enables users to train on arbitrary raw text documents without manual preprocessing. The smart chunking with overlap ensures coherent training samples while the format auto-detection simplifies the data loading workflow.
