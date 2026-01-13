# File: `tests/test_raw_text.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 172 |
| Classes | `MockDataset`, `MockTokenizer`, `MockTensor` |
| Functions | `test_raw_text_loader` |
| Imports | importlib, os, pathlib, sys, tempfile |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Minimal standalone test for the RawTextDataLoader and TextPreprocessor classes from the dataprep module, validating raw text training data preparation without requiring heavy dependencies.

**Mechanism:** The test creates mock implementations to avoid full dependency chains: (1) MockDataset mimics HF datasets.Dataset with column_names and dict-style access, (2) MockTokenizer simulates tokenization by word splitting with EOS token support and mock tensor output, (3) MockTensor provides basic tensor-like interface with tolist(). The test dynamically loads raw_text.py using importlib to avoid triggering unsloth package initialization. Test cases validate: text-based chunking output with 'text' column, tokenized output with 'input_ids', 'attention_mask', and 'labels' columns, chunk_size and stride parameter validation (positive chunk_size, stride < chunk_size), TextPreprocessor.clean_text whitespace normalization, and validate_dataset statistics generation. Cleanup removes the temporary test file.

**Significance:** This test validates the raw text training data pipeline, which enables training on plain text files rather than pre-formatted datasets. The lightweight mock approach allows testing the core chunking and preprocessing logic without GPU or model dependencies, making it suitable for CI environments and fast iteration. It ensures the foundational data preparation components work correctly before integration with full training pipelines.
