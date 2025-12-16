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

**Purpose:** Provides the SyntheticDataKit class for generating synthetic training data (question-answer pairs) from documents using a vLLM-backed inference server and LLM generation.

**Mechanism:** Core components:
- `PipeCapture`: Thread-based non-blocking pipe reader for capturing vLLM subprocess stdout/stderr with regex-based readiness detection and line buffering
- `terminate_tree()`: Cross-platform process termination using psutil for recursive tree killing or fallback to OS-specific commands
- `SyntheticDataKit`: Main class that:
  1. Spawns a vLLM server subprocess with configurable engine parameters (GPU utilization, KV cache settings, max sequence length)
  2. Monitors server startup via stdout/stderr capture and waits for readiness
  3. Provides `chunk_data()` to split large documents into overlapping chunks based on token limits
  4. Provides `prepare_qa_generation()` to configure YAML-based synthetic data generation pipeline (paths, prompts, cleanup thresholds)
  5. Implements cleanup methods to gracefully terminate vLLM server and free GPU memory
Uses context manager protocol (__enter__/__exit__) for automatic cleanup. Integrates with unsloth_zoo for vLLM patching and loading.

**Significance:** Enables automated generation of high-quality training datasets from raw documents without manual annotation. Critical for users who need to create custom fine-tuning datasets from their domain-specific documents. The vLLM integration provides high-throughput batch inference for efficient synthetic data generation at scale. The chunking and configuration system handles large documents that exceed model context windows.
