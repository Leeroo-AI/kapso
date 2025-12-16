# File: `unsloth/dataprep/synthetic.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 465 |
| Classes | `PipeCapture`, `SyntheticDataKit` |
| Functions | `terminate_tree` |
| Imports | collections, gc, numpy, os, re, requests, subprocess, synthetic_configs, threading, time, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides comprehensive toolkit for generating synthetic training data by managing a vLLM server process and orchestrating question-answer pair generation from source documents.

**Mechanism:** `PipeCapture` class implements non-blocking pipe capture with threading to monitor stdout/stderr of subprocess, supporting ready detection via regex and configurable buffering. `terminate_tree()` function ensures complete process cleanup using psutil or platform-specific commands. `SyntheticDataKit` main class: (1) initializes by loading model config/tokenizer, patching vLLM, spawning vLLM server subprocess with engine arguments, and waiting for ready signal via stdout monitoring, (2) `chunk_data()` method splits large documents into overlapping chunks sized to fit max_seq_length minus generation headroom, (3) `prepare_qa_generation()` creates output folder structure and generates YAML config file from template for QA generation pipeline. Includes context manager support (`__enter__`, `__exit__`) and cleanup via `terminate_tree()` plus vLLM module deletion.

**Significance:** Critical infrastructure for data augmentation workflows, particularly for fine-tuning on domain-specific data where labeled examples are scarce. The vLLM integration enables high-throughput generation while the chunking logic ensures long documents can be processed without hitting context limits. The subprocess management with robust cleanup prevents GPU memory leaks. The YAML config generation bridges to external tools for multi-stage data preparation pipelines (ingest -> generate -> clean -> format). The overlapping chunks maintain context continuity across boundaries. Setting `HF_HUB_ENABLE_HF_TRANSFER=1` optimizes model downloads.
