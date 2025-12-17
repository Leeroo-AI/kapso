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

**Purpose:** Framework for generating synthetic QA training data using vLLM-powered LLM inference with data chunking, cleanup, and multiprocessing support.

**Mechanism:** Implements SyntheticDataKit class that: (1) spawns vLLM server as subprocess with custom engine args and monitors readiness; (2) provides chunk_data() to split texts into overlapping chunks respecting token limits; (3) prepares QA generation configs via prepare_qa_generation() method. Uses PipeCapture class for non-blocking stdout/stderr monitoring with regex-based readiness detection.

**Significance:** Enables large-scale synthetic training data generation for fine-tuning, automating QA pair creation from unstructured text while managing vLLM subprocess lifecycle.
