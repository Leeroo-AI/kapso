# File: `utils/download_glue_data.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 160 |
| Functions | `download_and_extract`, `format_mrpc`, `download_diagnostic`, `get_tasks`, `main` |
| Imports | argparse, os, sys, urllib, zipfile |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Downloads and prepares GLUE benchmark datasets for NLP model evaluation. Handles data retrieval, extraction, and formatting for tasks like CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI, and diagnostic datasets.

**Mechanism:** Fetches zip files from Firebase storage URLs, extracts them to specified directories. Special handling for MRPC dataset which requires processing train/test/dev splits using provided ID files or downloading from Facebook SentEval. Downloads diagnostic dataset as single TSV file. Command-line interface accepts task names and data directory path.

**Significance:** Standard utility for researchers and practitioners testing models on GLUE benchmarks. Simplifies data acquisition for a widely-used evaluation suite. Addresses MRPC licensing complications by supporting multiple sources and providing manual extraction instructions for Windows/Mac/Linux users.
