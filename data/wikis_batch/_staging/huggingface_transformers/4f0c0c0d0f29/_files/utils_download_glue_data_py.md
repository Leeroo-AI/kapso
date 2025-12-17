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

**Purpose:** Downloads and formats GLUE benchmark datasets for NLP evaluation tasks.

**Mechanism:** Downloads 11 GLUE tasks (CoLA, SST, MRPC, QQP, STS, MNLI, SNLI, QNLI, RTE, WNLI, diagnostic) from Firebase storage URLs. Special handling for MRPC: downloads train/test files from Facebook's SentEval, retrieves dev_ids.tsv, and splits training data into train/dev sets based on those IDs. Formats data into .tsv files with consistent headers for downstream consumption.

**Significance:** Data acquisition utility for researchers and developers working with GLUE benchmark, providing standardized dataset downloads with proper train/dev/test splits. Note: MRPC requires special handling due to licensing restrictions on direct hosting.
