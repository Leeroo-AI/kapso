# File: `unsloth/dataprep/raw_text.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 348 |
| Classes | `RawTextDataLoader`, `TextPreprocessor` |
| Imports | csv, datasets, json, os, pathlib, re, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Convert raw text files into tokenized training datasets for causal language modeling

**Mechanism:** RawTextDataLoader handles multiple formats (.txt, .md, .json, .jsonl, .csv) with auto-detection, implements smart_chunk_text() that tokenizes entire documents then chunks with configurable stride overlap, returns datasets with input_ids/attention_mask/labels ready for training. TextPreprocessor provides text cleaning, section extraction, structure token addition, and dataset validation with quality statistics

**Significance:** Essential data preparation utility that bridges the gap between raw documents and training-ready datasets, enabling efficient processing of various text formats with proper tokenization, overlap handling for context preservation, and quality validation to ensure clean training data
